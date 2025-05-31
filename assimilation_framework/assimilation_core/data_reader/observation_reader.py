import numpy as np
import torch

from assimilation_framework.utils.misc import load_config

class ObservationReader:
    def __init__(self, obs_type, da_win, config):
        self.type       = obs_type
        self.da_win     = da_win
        self.config     = load_config(config)

    def add_noise(self, field, noise_level):
        pass

    def get_obs_from_simulation(self, gt):
        if self.config["sampling"] == "random":
            matrix = np.zeros((gt.shape[2], gt.shape[3]), dtype=int)
            indices = np.random.choice(matrix.size, self.config["density"], replace=False)

            np.put(matrix, indices, 1)

            x = np.stack([matrix] * gt.shape[1])
            mask = torch.from_numpy(np.stack([x] * gt.shape[0])).to(torch.float32)

        elif self.config["sampling"] == "fixed":
            x = np.load(self.config["mask_path"])
            mask = torch.from_numpy(np.stack([x] * gt.shape[0])).to(torch.float32)

        else:
            raise NotImplementedError

        if self.config["add_noise"]:
            pass

        return gt * mask, mask

    def get_obs(self, tstamp, gt=None):
        if self.type == "simulation":
            obs, mask = self.get_obs_from_simulation(gt)
            return obs, mask
        elif self.type == "gdas":
            return NotImplementedError
        else:
            return NotImplementedError
