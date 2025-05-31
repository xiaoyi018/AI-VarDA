import json
from collections import OrderedDict

import yaml
import torch
import numpy as np

from assimilation_framework.networks.fengwu_hr.LGUnet_all import LGUnet_all_1

class ForecastModel:
    def __init__(self, config, device):
        self.name        = config["name"]
        self.conf_path   = config["config"]
        self.ckpt_path   = config["checkpoint"]
        self.device      = device

        with open(config["mean_std"], 'r') as file:
            self.mean_std = json.load(file)
        self.mean  = torch.from_numpy(np.array(self.mean_std["mean"])).to(torch.float32).to(self.device)
        self.std   = torch.from_numpy(np.array(self.mean_std["std"])).to(torch.float32).to(self.device)

        self.model = self.init_model()
        self.load_checkpoint()

    def init_model(self):
        if self.name == "fengwu_hr_6h":
            with open(self.conf_path, 'r') as cfg_file:
                cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
            model = LGUnet_all_1(**cfg_params["model"]["params"]["sub_model"]["lgunet_all"])
        else:
            raise NotImplementedError

        return model

    def normalize(self, input):
        if self.name == "fengwu_hr_6h":
            return (input - self.mean.reshape(-1, 1, 1)) / self.std.reshape(-1, 1, 1)

        else:
            raise NotImplementedError
    
    def denormalize(self, input):
        if self.name == "fengwu_hr_6h":
            return input * self.std.reshape(-1, 1, 1) + self.mean.reshape(-1, 1, 1)

        else:
            raise NotImplementedError

    def load_checkpoint(self):
        checkpoint_dict  = torch.load(self.ckpt_path)
        checkpoint_model = checkpoint_dict['model']['lgunet_all']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)

    def do_forecast(self, input, step):
        input = input.to(self.device)
        z = self.normalize(input).unsqueeze(0)

        for i in range(step):
            z = self.model(z)[:, :input.shape[0]].detach()

        return self.denormalize(z.reshape(*input.shape)).cpu()
