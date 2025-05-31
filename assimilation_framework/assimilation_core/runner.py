"""
Main class for running cyclic data assimilation.
"""
import os
import time
import functools

import pandas as pd
import numpy as np
import torch
from petrel_client.client import Client

from assimilation_framework.assimilation_core.data_reader.field_reader import FieldReader
from assimilation_framework.assimilation_core.data_reader.observation_reader import ObservationReader
from assimilation_framework.assimilation_core.forecast_model import ForecastModel
from assimilation_framework.assimilation_core.init_states_constructor import InitStatesConstructor
from assimilation_framework.assimilation_core.ensemble_generator import EnsembleGenerator
from assimilation_framework.assimilation_core.assimilation_models import DAModelWrapper
from assimilation_framework.assimilation_core.observation_operator import ObservationOperatorBuilder
from assimilation_framework.assimilation_core.flow_models import flow_model_builder
from assimilation_framework.assimilation_core.evaluator import evaluate_from_metric
from assimilation_framework.utils.metrics import Metrics
from assimilation_framework.utils.misc import FieldWriter

class AssimilationRunner:
    def __init__(self, config, run_dir, logger):
        self.config     = config
        self.run_dir    = run_dir
        self.logger     = logger
        self.metrics    = Metrics()
        self.client     = Client(conf_path=config["client_path"])
        self.save_dir_type = config["save"]["field"]["path_config"]["type"]
        # self.field_writer = FieldWriter(self.save_dir_type, self.client)
        self.save_dir_root = os.path.join(config["save"]["field"]["path_config"]["path"], "intermediate_ens", config["date"], config["prefix"])

        self.start_time = pd.Timestamp(config["start_time"])
        self.end_time   = pd.Timestamp(config["end_time"])
        self.cycle_time = pd.Timedelta(config["cycle_time"])
        self.flow_cycle_time = pd.Timedelta(config["flow_cycle_time"])
        self.da_win     = config["da_win"]

        self.init_directory()
        self.current_time = self.start_time
        self.metrics_list = {key: [] for key in self.config["save"]["eval_results"]["metrics_list"]}
        self.field_reader       = FieldReader(field_type = config["ground_truth"], client = self.client)
        self.observation_reader = ObservationReader(obs_type = config["observation"]["type"], da_win = self.da_win, config = config["observation"]["config"])

        self.forecast_model  = ForecastModel(config = config["forecast"]["forecast_model"], device = config["device"])
        self.init_states_constructor = InitStatesConstructor(config = config["initial_states"], client = self.client, save_path = self.save_dir_root, forecast_model = self.forecast_model, field_reader = self.field_reader, tstamp = self.start_time, dtime = self.cycle_time)
        self.ensemble_generator = EnsembleGenerator(config = config["forecast"]["ensemble_gen"], client = self.client, load_path = self.save_dir_root, forecast_model = self.forecast_model)
        self.flow_model, self.flow_error = flow_model_builder(config["da_method"]["flow_model"], device = config["device"], logger=self.logger, da_win=self.da_win)
        self.da_model_wrapper = DAModelWrapper(config["da_method"]["framework"], device = config["device"], logger=self.logger, da_win=self.da_win, flow_model=self.flow_model, flow_error=self.flow_error, path = self.ckpts_dir)
        self.obs_op_builder = ObservationOperatorBuilder(config["da_method"]["observation_operator"], da_win=self.da_win)

    def init_directory(self):
        self.eval_metrics_dir    = os.path.join(self.run_dir, "eval_metrics")
        self.physical_fields_dir = os.path.join(self.run_dir, "physical_fields")
        self.ckpts_dir           = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.eval_metrics_dir, exist_ok=True)
        os.makedirs(self.physical_fields_dir, exist_ok=True)
        os.makedirs(self.ckpts_dir, exist_ok=True)

    def resume_eval_results(self):
        for key in self.config["save"]["eval_results"]["metrics_list"]:
            path = os.path.join(self.eval_metrics_dir, f"{key}.npy")
            if os.path.exists(path):
                self.metrics_list[key] = np.load(path).tolist()

    def resume_da_system(self):
        path = os.path.join(self.run_dir, "current_time.txt")
        if os.path.exists(path):
            f = open(path, "r")
            self.current_time = pd.Timestamp(f.read())
            self.da_model_wrapper.resume()
            self.resume_eval_results()
        else:
            self.init_states_constructor.construct_init_states()

    def save_fields(self, xa, yo, gt, tstamp):
        if self.config["save"]["field"]["analysis"]:
            np.save(os.path.join(self.physical_fields_dir, "xa.npy"), xa.numpy())
        if self.config["save"]["field"]["observation"]:
            np.save(os.path.join(self.physical_fields_dir, "obs.npy"), yo.numpy())
        if self.config["save"]["field"]["ground_truth"]:
            np.save(os.path.join(self.physical_fields_dir, "gt.npy"), gt.numpy())

    def save_eval_results(self):
        for key in self.config["save"]["eval_results"]["metrics_list"]:
            path = os.path.join(self.eval_metrics_dir, f"{key}.npy")
            np.save(path, self.metrics_list[key])

    def _save_da_system(self):
        with open(os.path.join(self.run_dir, "current_time.txt"), 'w') as f:
            f.write(str(self.current_time))
        self.da_model_wrapper.save()
        self.save_eval_results()

    def eval_results(self, xb, xa, gt):
        for key in self.config["save"]["eval_results"]["metrics_list"]:
            self.metrics_list[key].append(evaluate_from_metric(xb, xa, gt, key, self.metrics))

    def run_cycle(self, timestamp):

        @self.timeit("Updating bg")
        def update_da_model(ensemble): 
            self.da_model_wrapper.update(xb_ensemble)

        @self.timeit("DA")
        def assimilate(xb, yo, obs_op_set, obs_err, gt0):
            return self.da_model_wrapper.assimilate(xb, yo, obs_op_set, obs_err, gt0)

        @self.timeit("Ensemble integration")
        def gen_bg_ensemble(xa):
            self.ensemble_generator.gen_bg_ensemble(xa, self.current_time, self.cycle_time)

        xb_ensemble, xb = self.ensemble_generator.get_bg_ensemble(timestamp)
        gt  = torch.stack([self.field_reader.get_state(timestamp + i * self.flow_cycle_time) for i in range(self.da_win)])
        gt0 = gt[0]
        yo, mask = self.observation_reader.get_obs(timestamp, gt)
        obs_op_set = {"mask": mask, "op": self.obs_op_builder.obs_op}
        obs_err = self.obs_op_builder.get_obs_var(yo, gt)
        update_da_model(xb_ensemble)
        xa = assimilate(xb, yo, obs_op_set, obs_err, gt0)

        self.save_fields(xa, yo, gt[0], timestamp)
        self.eval_results(xb.unsqueeze(0), xa.unsqueeze(0), gt0.unsqueeze(0))

        gen_bg_ensemble(xa)


    def run(self):

        @self.timeit("Saving DA parameters")
        def save_da_system():
            self._save_da_system()

        if self.config["resume"]:
            self.resume_da_system()

        while(self.current_time + self.cycle_time <= self.end_time):
            self.logger.info(f"current time: {self.current_time}")

            self.run_cycle(self.current_time)
            self.current_time += self.cycle_time

            save_da_system()

    def timeit(self, label):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                if self.logger:
                    self.logger.info(f"{label} finished. Time consumed: {end - start:.2f} (s)")
                return result
            return wrapper
        return decorator
