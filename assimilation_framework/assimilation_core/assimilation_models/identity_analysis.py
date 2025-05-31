"""
No assimilation (analysis equals background)
"""

from .base_assimilation import AssimilationModel

class IdentityAnalysis(AssimilationModel):
    def assimilate(self, background, observations, obs_op_set, obs_err, gt):
        """
        Return background directly as the analysis.
        """
        self.logger.info("Running identity assimilation")
        
        return background

