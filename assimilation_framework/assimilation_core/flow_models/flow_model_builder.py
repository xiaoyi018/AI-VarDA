import numpy as np
import torch

from .fengwu_hr2hr_flow import FengWuHR2HRFlow
from assimilation_framework.utils.misc import load_config

def get_flow_error(err_type, err_dim, da_win):
    if err_type == 0:
        q = torch.zeros(da_win, *err_dim, requires_grad=True, device=self.device)
    elif err_type == 1:
        q = torch.broadcast_to(torch.from_numpy(np.load("checkpoints/flow_error/new_q.npy")).unsqueeze(2).unsqueeze(2)[:da_win-1], (da_win-1, *err_dim))
    else:
        return NotImplementedError
    return q

def flow_model_builder(config, device, logger, da_win):
    if config["type"] == "none":
        assert da_win > 1, "when da_win > 1, flow model should be defined!"
        flow_model = None
        flow_error = None
    elif config["type"] == "hr2hr":
        flow_model = FengWuHR2HRFlow(config, device, logger)
        flow_error = get_flow_error(config["err_type"], config["err_dim"], da_win).to(device)
    else:
        raise NotImplementedError

    return flow_model, flow_error