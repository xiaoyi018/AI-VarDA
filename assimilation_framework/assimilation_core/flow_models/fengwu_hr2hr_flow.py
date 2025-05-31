import json
from collections import OrderedDict

import yaml
import torch
import numpy as np

from .base_flow_model import FlowModel
from assimilation_framework.networks.fengwu_hr.LGUnet_all import LGUnet_all_1

class FengWuHR2HRFlow(FlowModel):

    def __init__(self, config, device, logger):
        self.params = config["params"]
        self.ckpts  = config["checkpoint"]
        self.logger = logger
        self.device = device
        self.model  = self.init_model_flow()
        with open(config["model_mean_std"], 'r') as file:
            self.mean_std = json.load(file)
        self.mean  = torch.from_numpy(np.array(self.mean_std["mean"])).float().to(self.device)
        self.std   = torch.from_numpy(np.array(self.mean_std["std"])).float().to(self.device)

    def normalize(self, input):
        return (input - self.mean.reshape(-1, 1, 1)) / self.std.reshape(-1, 1, 1)
    
    def denormalize(self, input):
        return input * self.std.reshape(-1, 1, 1) + self.mean.reshape(-1, 1, 1)

    def init_model_flow(self):
        with open(self.params, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        model = LGUnet_all_1(**cfg_params["model"]["params"]["sub_model"]["lgunet_all"])
        checkpoint_dict = torch.load(self.ckpts)
        checkpoint_model = checkpoint_dict['model']['lgunet_all']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def do_flow(self, input):
        z = self.normalize(input).unsqueeze(0)

        z = self.model(z)[:, :input.shape[0]]

        return self.denormalize(z.reshape(*input.shape))


    
