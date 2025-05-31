import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_harmonics import RealSHT, InverseRealSHT

from .base_decoder import BackgroundDecoder

class GenBeDecoder(BackgroundDecoder):
    """
    Simple linear mapping from control variable z to background state x.
    Often used in classical variational assimilation.
    Channels: suf. var. (4) + upper-air var. (nlevel) * 5
    """
    def __init__(self, config, device, logger=None):
        super().__init__()
        self.device    = device
        self.logger    = logger
        self.nlat_in   = config["nlat_in"]
        self.nlat_out  = config["nlat_out"]
        self.nlon_in   = config["nlon_in"]
        self.nlon_out  = config["nlon_out"]
        self.nchn_in   = config["nchn_in"]
        self.nchn_out  = config["nchn_out"]
        self.nlevel    = config["nlevel"]
        self.radius    = config["radius"]
        self.hpad      = config["hpad"]
        self.scale_factor = config["scale_factor"]
        self.magnify      = config["magnify"]
        self.bmatrix_path = config.get("bmatrix_path", None)
        
        if self.bmatrix_path is not None:
            self.len_scale = torch.from_numpy(np.load(os.path.join(config["bmatrix_path"], "len_scale.npy")) * self.scale_factor).float().to(self.device)
            self.reg_coeff = torch.from_numpy(np.load(os.path.join(config["bmatrix_path"], "reg_coeff.npy"))).float().to(self.device)
            self.std_sur   = torch.from_numpy(np.load(os.path.join(config["bmatrix_path"], "std_sur.npy"))).float().to(self.device)
            self.vert_eig_value = torch.from_numpy(np.load(os.path.join(config["bmatrix_path"], "vert_eig_value.npy"))).float().to(self.device)
            self.vert_eig_vec   = torch.from_numpy(np.load(os.path.join(config["bmatrix_path"], "vert_eig_vec.npy"))).float().to(self.device)
        else:
            self.len_scale = torch.from_numpy(config["len_scale"] * self.scale_factor).float().to(self.device)
            self.reg_coeff = torch.from_numpy(config["reg_coeff"]).float().to(self.device)
            self.std_sur   = torch.from_numpy(config["std_sur"]).float().to(self.device)
            self.vert_eig_value = torch.from_numpy(config["vert_eig_value"]).float().to(self.device)
            self.vert_eig_vec   = torch.from_numpy(config["vert_eig_vec"]).float().to(self.device)

        self.get_additional_info()

    def get_additional_info(self):
        self.sht = RealSHT(self.nlat_in, self.nlon_in, grid="equiangular").to(self.device)
        self.isht = InverseRealSHT(self.nlat_in, self.nlon_in, grid="equiangular").to(self.device)

        kernel = torch.zeros(self.nchn_in, self.nlat_in, self.nlon_in).to(self.device)
        self.coeffs_kernel = []
        for layer in range(self.nchn_in):
            for i in range(self.hpad):
                kernel[layer, i] = torch.exp(-i**2/(8*self.len_scale[layer]**2))
            self.coeffs_kernel.append(self.sht(kernel[layer]))

        self.sph_scale = torch.Tensor(np.array(np.broadcast_to(np.arange(0, self.nlat_in).transpose(), (self.nlat_in+1, self.nlat_in)).transpose())).to(self.device)
        self.sph_scale = 2*np.pi*torch.sqrt(4*np.pi/(2*self.sph_scale+1))

    def decode(self, z):
        inc_static = []

        for i in range(self.nchn_in):
            coeffs_field  = self.sht(z[i])
            inc_static.append(self.isht(self.sph_scale*coeffs_field*self.coeffs_kernel[i][:, 0].reshape((self.nlat_in, 1))).unsqueeze(0))

        inc_static = torch.cat(inc_static, 0)
        inc_static = self.magnify * inc_static / (self.len_scale.reshape(-1, 1, 1)**2)

        inc_psi   = inc_static[4+self.nlevel*2:4+self.nlevel*3]
        inc_vmode = torch.clone(inc_static)

        for i in range(self.nchn_in):
            inc_vmode[i] = inc_static[i] + torch.sum(inc_psi * self.reg_coeff[i].reshape(-1, 1, 1), 0)

        inc_sfvp  = torch.clone(inc_vmode)
        inc_sfvp[0:4] = inc_vmode[0:4] * self.std_sur.reshape(-1, 1, 1)

        for i in range(5):
            sample = inc_vmode[4+self.nlevel*i:4+self.nlevel*(i+1), :, :].reshape(self.nlevel, -1)
            sample = torch.matmul(self.vert_eig_vec[i], torch.matmul(torch.sqrt(torch.diag(self.vert_eig_value[i])), sample)).reshape(self.nlevel, self.nlat_in, self.nlon_in)
            inc_sfvp[4+self.nlevel*i:4+self.nlevel*(i+1), :, :] = sample

        def partial_x(field):
            x_scaling = torch.sin(torch.linspace(1/180 * torch.pi, 179/180 * torch.pi, self.nlat_in)).reshape(1, -1, 1).to(self.device)
            field_shift_1 = torch.cat([field[:, :,  1:], field[:, :,  :1]], 2)
            field_shift_2 = torch.cat([field[:, :, -1:], field[:, :, :-1]], 2)
            return (field_shift_2 - field_shift_1) / (2 * self.radius * 180 / self.nlat_in * x_scaling)

        def partial_y(field):
            lat_coord = (torch.arange(self.nlat_in).to(self.device) * self.radius * 180 / (self.nlat_in-1), )
            return torch.gradient(field, spacing = lat_coord, dim=1)[0]

        inc_recon  = torch.clone(inc_sfvp)

        sfx = partial_x(inc_sfvp[4+self.nlevel*2:4+self.nlevel*3])
        sfy = partial_y(inc_sfvp[4+self.nlevel*2:4+self.nlevel*3])
        vpx = partial_x(inc_sfvp[4+self.nlevel*3:4+self.nlevel*4])
        vpy = partial_y(inc_sfvp[4+self.nlevel*3:4+self.nlevel*4])

        inc_recon[4+self.nlevel*2:4+self.nlevel*3] =  sfy - vpx
        inc_recon[4+self.nlevel*3:4+self.nlevel*4] = -sfx - vpy

        output = F.interpolate(inc_recon.unsqueeze(0), (self.nlat_out, self.nlon_out)).squeeze(0)

        return output
