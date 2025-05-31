import copy
import os

from .base_assimilation import AssimilationModel
from .identity_analysis import IdentityAnalysis
from .var_assimilation import VarAssimilation
from assimilation_framework.utils.misc import load_config
from assimilation_framework.assimilation_core.bg_decoder_models import GenBeDecoder, VAEDecoder
from assimilation_framework.bgerr_learning.learning_algorithms import genbe_low_dim

class DAModelWrapper:
    def __init__(self, config, device, logger, da_win, flow_model, flow_error, path):
        self.type      = config["type"]
        self.da_config = load_config(config["config"])
        self.device = device
        self.logger = logger
        self.da_win = da_win
        self.flow_model = flow_model
        self.flow_error = flow_error
        self.save_path  = os.path.join(path, "bg_decoder_ckpt")
        self.da_model = self.init_da_model()

    def init_da_model(self,):
        if self.type == "no_da":
            model = IdentityAnalysis(device=self.device, logger=self.logger)

        elif self.type == "gaussian":
            bg_decoder = GenBeDecoder(config=self.da_config, device=self.device, logger=self.logger)
            coeff_dyn = self.da_config.get("coeff_dyn", 0.0)
            model = VarAssimilation(device=self.device, logger=self.logger, nchn_in=self.da_config["nchn_in"], nlat_in=self.da_config["nlat_in"], nlon_in=self.da_config["nlon_in"], bg_lamb=self.da_config["bg_lamb"], nit=self.da_config["niter"], maxit=self.da_config["maxiter"], da_win=self.da_win, bg_decoder=bg_decoder, coeff_dyn=self.da_config["coeff_dyn"], flow_model=self.flow_model, flow_error=self.flow_error)

        elif self.type == "vae":
            bg_decoder = VAEDecoder(config=self.da_config, device=self.device, logger=self.logger)
            model = VarAssimilation(device=self.device, logger=self.logger, nchn_in=self.da_config["nchn_in"], nlat_in=self.da_config["nlat_in"], nlon_in=self.da_config["nlon_in"], bg_lamb=self.da_config["bg_lamb"], nit=self.da_config["niter"], maxit=self.da_config["maxiter"], da_win=self.da_win, bg_decoder=bg_decoder, flow_model=self.flow_model, flow_error=self.flow_error)

        else:
            raise NotImplementedError

        return model

    def update(self, xb_ensemble):
        if len(xb_ensemble) < 2: # 4
            self.logger.info(f"Ensemble size: {len(xb_ensemble)}. Background information not updated.")
            return
        else:
            self.logger.info(f"Ensemble size: {len(xb_ensemble)}. Updating background information.")
            if self.type == "gaussian":
                flow_bg = genbe_low_dim(xb_ensemble)
                bg_config_dyn = copy.deepcopy(self.da_config)
                bg_config_dyn["bmatrix_path"] = None
                bg_config_dyn.update(flow_bg)
                bg_decoder_dyn = GenBeDecoder(config=bg_config_dyn, device=self.device, logger=self.logger)
                self.da_model.bg_decoder_dyn = bg_decoder_dyn
            elif self.type == "vae":
                if self.da_config["finetune_type"] == "no_tune":
                    pass
                elif self.da_config["finetune_type"] == "full_finetune" or self.da_config["finetune_type"] == "reset_finetune" or self.da_config["finetune_type"] == "lora_finetune":
                    self.da_model.bg_decoder.trainer(xb_ensemble, lr=float(self.da_config["learning_rate"]), epoch=self.da_config["finetune_epoch"], bs=self.da_config["batch_size"], finetune_type=self.da_config["finetune_type"])
                else:
                    raise NotImplementedError
            elif self.type == "no_da":
                pass
            return 

    def assimilate(self, xb, yo, obs_op_set, obs_err, gt0):
        return self.da_model.assimilate(xb, yo, obs_op_set, obs_err, gt0)

    def save(self,):
        if self.type == "vae":
            if self.da_config["finetune_type"] == "full_finetune":
                self.da_model.bg_decoder.save_checkpoint(self.save_path)
            elif self.da_config["finetune_type"] == "lora_finetune":
                self.da_model.bg_decoder.save_lora_checkpoint(self.save_path)
            else:
                pass
        else:
            pass

    def resume(self,):
        if self.type == "vae":
            if self.da_config["finetune_type"] == "full_finetune" or self.da_config["finetune_type"] == "lora_finetune":
                if os.path.exists(self.save_path):
                    self.da_model.bg_decoder.load_checkpoint(self.save_path)
            else:
                pass
        else:
            pass
        