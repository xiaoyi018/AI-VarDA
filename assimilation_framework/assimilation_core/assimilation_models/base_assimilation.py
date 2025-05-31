"""
Base class for assimilation models.
"""

from assimilation_framework.utils.metrics import Metrics

class AssimilationModel:
    def __init__(self, device="cpu", logger=None):
        self.device     = device
        self.logger     = logger
        self.metric     = Metrics()

    def to_device(self, tensor):
        if isinstance(tensor, list):
            return [t.to(self.device) for t in tensor]
        elif isinstance(tensor, dict):
            output = {}
            for k, v in tensor.items():
                try:
                    v = v.to(self.device)
                except:
                    pass
                output[k] = v
            return output
            # return {k: v.to(self.device) for k, v in tensor.items()}
        else:
            return tensor.to(self.device)

    def assimilate(self, background, observations, obs_op_set, obs_err, gt):
        """
        obs_op_set includes mask and obs_op
        gt for evaluation only
        """
        raise NotImplementedError
