import json

import numpy as np
import torch

from .identity_operator import IdentityObservationOperator
from .pressure_interp_operator import PressureLevelOperator

class ObservationOperatorBuilder:
    def __init__(self, config, da_win):
        self.type        = config["type"]
        self.cov_type    = config["err_cov_type"]
        self.err_std     = config["err_std"]
        self.modify_type = config["modify_type"]
        with open(config["general_mean_std"], 'r') as file:
            self.mean_std = json.load(file)
        self.da_win   = da_win
        self.obs_op   = self.get_obs_op()

    def get_obs_op(self,):
        if self.type == "identity":
            obs_op = lambda x: x

        else:
            raise NotImplementedError

        return obs_op

    def get_obs_var(self, obs, gt=None):
        if self.cov_type == "predefined":
            obs_var = np.array(self.mean_std["std"])**2 * self.err_std**2
            R = torch.zeros(*obs.shape) + obs_var.reshape(1, -1, 1, 1)
            if self.modify_type == "modify_temperature":
                R[:, 56:69] /= 16
                R[:, 2] /= 16
            return R

        else:
            raise NotImplementedError
