"""
Variational Assimilation (including Gaussian-based 3DVar/4DVar and VAE-based 3DVar/4DVar)
"""

import numpy as np
import torch
import torch.utils.checkpoint as cp
import torch.optim as optim

from .base_assimilation import AssimilationModel

class VarAssimilation(AssimilationModel):
    def __init__(self, nchn_in, nlat_in, nlon_in, bg_lamb, nit, maxit, da_win, bg_decoder, bg_decoder_dyn=None, coeff_dyn=0.0, flow_model=None, flow_error=None, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = (nchn_in, nlat_in, nlon_in)
        self.bg_lamb    = bg_lamb
        self.Nit        = nit
        self.maxit      = maxit
        self.da_win     = da_win
        self.bg_decoder     = bg_decoder
        self.bg_decoder_dyn = bg_decoder_dyn
        self.coeff_static   = 1 - coeff_dyn # np.sqrt(1 - coeff_dyn**2)
        self.coeff_dyn      = coeff_dyn
        self.flow_model     = flow_model
        self.flow_error     = flow_error

        if self.da_win > 1:
            if self.flow_model is None:
                raise ValueError("flow_model must be provided when da_win > 1 (4DVar assimilation)")
            if self.flow_error is None:
                raise ValueError("flow_error must be provided when da_win > 1 (4DVar assimilation)")
            if not self.flow_error.shape[0] == self.da_win - 1:
                raise ValueError("flwo error shape does not match window size.")

    def transform(self, z, xb):
        if self.bg_decoder_dyn is None:
            return self.bg_decoder.decode(z) + xb
        else:
            return self.coeff_static * self.bg_decoder.decode(z[0]) + self.coeff_dyn * self.bg_decoder_dyn.decode(z[1]) + xb

    def assimilate(self, background, observations, obs_op_set, obs_err, gt):
        
        self.logger.info("Running variational assimilation")

        assert observations.shape[0] == self.da_win, \
            f"observations has {observations.shape[0]} time steps, expected {self.da_win}"

        xb = self.to_device(background)
        yo = self.to_device(observations)
        obs_op_set = self.to_device(obs_op_set)
        obs_err = self.to_device(obs_err)
        gt = self.to_device(gt)

        obs_flow_err = torch.clone(obs_err)
        obs_flow_err[1:] += self.flow_error

        def cal_loss_bg(z):
            """
            z:     C x H x W
            """
            return torch.sum(z**2) / 2

        def cal_loss_obs(x, y, mask, obs_op, rq):
            """
            x:       C x H x W
            y:       T x C x H x W
            R:       T x C x H x W
            """
            x_list = [x, ]
            for i in range(self.da_win - 1):
                x = cp.checkpoint(lambda x_: self.flow_model.do_flow(x_), x)
                x_list.append(x)

            x_pred = torch.stack(x_list, 0)   # T x C x H x W

            return torch.sum( mask * (obs_op(x_pred) - yo) ** 2 / rq ) / 2

        def cal_loss(z):
            loss_bg = torch.sum(z**2) / 2
            x = self.transform(z, xb)
            loss_obs = cal_loss_obs(x, yo, obs_op_set["mask"], obs_op_set["op"], obs_flow_err)
            return loss_bg + loss_obs / self.bg_lamb

        if self.bg_decoder_dyn is None:
            z = torch.zeros(*self.latent_dim, requires_grad=True, device=self.device)
        else:
            z = torch.zeros(2, *self.latent_dim, requires_grad=True, device=self.device)
        lbfgs = optim.LBFGS([z], history_size=10, max_iter=self.maxit, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            objective = cal_loss(z)
            objective.backward()
            return objective 
    
        for kk in range(self.Nit+1):
            xhat = self.transform(z, xb)
            wrmse = self.metric.WRMSE(xhat.unsqueeze(0).clone().detach().cpu(), gt.unsqueeze(0).clone().detach().cpu(), None, None, 1).detach()
            bias = self.metric.Bias(xhat.unsqueeze(0).clone().detach().cpu(), gt.unsqueeze(0).clone().detach().cpu(), None, None, 1).detach()
            loss_bg = cal_loss_bg(z).detach()
            loss_obs = cal_loss_obs(xhat, yo, obs_op_set["mask"], obs_op_set["op"], obs_flow_err).detach()
            loss = loss_bg + loss_obs / self.bg_lamb
            
            if self.logger:
                self.logger.info(f"[Iter {kk}] loss bg: {loss_bg} loss obs: {loss_obs} loss: {loss}")
                self.logger.info(f"[Iter {kk}] RMSE (z500): {wrmse[11].item():.4g} Bias (z500): {bias[11].item():.4g} q500: {wrmse[24].item():.4g}, t2m: {wrmse[2].item():.4g} t850: {wrmse[66].item():.4g} u500: {wrmse[37].item():.4g}, v500: {wrmse[50].item():.4g}")
            
            if kk < self.Nit:
                torch.cuda.empty_cache()
                lbfgs.step(closure)

        return xhat.detach().cpu()
