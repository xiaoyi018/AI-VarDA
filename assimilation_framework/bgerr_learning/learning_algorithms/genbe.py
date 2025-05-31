import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from xspharm import xspharm
from spharm import Spharmt
from petrel_client.client import Client

class GenBeAgent:
    def __init__(self, client, error_path, output_path, lat, lon):
        self.client = client
        self.output_path = output_path
        self.lat = lat
        self.lon = lon

        with io.BytesIO(self.client.get(error_path)) as f:
            self.error_sample = np.load(f)

        # self.error_sample = np.load(error_path)

        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        bmatrix = genbe_low_dim(self.error_sample, self.lat, self.lon)
        for k in bmatrix:
            np.save(os.path.join(self.output_path, k), bmatrix[k])

def genbe_low_dim(data, lat=128, lon=256):
    if isinstance(data, list): # for finetuning, data is list of torch.tensors
        data = torch.stack(data).numpy()
    mean = np.mean(data, axis=0)
    err_sample = data - mean

    b_matrix = {}

    err_sample = F.interpolate(torch.Tensor(err_sample), (lat, lon)).numpy()
    err_sample_sfvp = np.copy(err_sample)
    lons = np.linspace(0, 360, lon, endpoint=False)
    lats = np.linspace(89.9, -89.9, lat)
    for idx in range(err_sample_sfvp.shape[0]):
        ds_uv = xr.Dataset(
                data_vars=dict(
                    u = (['lev', 'lat', 'lon'],  err_sample[idx, 4+13*2:4+13*3]), 
                    v = (['lev', 'lat', 'lon'],  err_sample[idx, 4+13*3:4+13*4]), 
                ),
                coords=dict(
                    lat =(["lat"], lats),
                    lon =(["lon"], lons),
                ),
            )

        Xsp = xspharm(ds_uv)
        sfvp_ds = Xsp.uv2sfvp(ds_uv.u, ds_uv.v)

        err_sample_sfvp[idx, 4+13*2:4+13*3] = sfvp_ds.sf
        err_sample_sfvp[idx, 4+13*3:4+13*4] = sfvp_ds.vp

    error_sur = err_sample_sfvp[:, 0:4, :, :]
    error_z   = err_sample_sfvp[:, 4+13*0:4+13*1, :, :]
    error_q   = err_sample_sfvp[:, 4+13*1:4+13*2, :, :]
    error_psi = err_sample_sfvp[:, 4+13*2:4+13*3, :, :]
    error_chi = err_sample_sfvp[:, 4+13*3:4+13*4, :, :]
    error_t   = err_sample_sfvp[:, 4+13*4:4+13*5, :, :]

    std_sur = np.sqrt(np.mean(np.var(error_sur, 0), (1, 2)))

    var_z   = np.mean(np.var(error_z, 0), (1, 2))
    var_q   = np.mean(np.var(error_q, 0), (1, 2))
    var_psi = np.mean(np.var(error_psi, 0), (1, 2))
    var_chi = np.mean(np.var(error_chi, 0), (1, 2))
    var_t   = np.mean(np.var(error_t, 0), (1, 2))
    
    def cal_cov(var_arr):
        res = np.zeros((var_arr.shape[0], var_arr.shape[0]))
        for i in range(var_arr.shape[0]):
            for j in range(var_arr.shape[0]):
                res[i, j] = np.sqrt(var_arr[i]*var_arr[j]) * np.exp(-np.abs(i-j))
                
        return res
                
    cov_z   = cal_cov(var_z)
    cov_q   = cal_cov(var_q)
    cov_psi = cal_cov(var_psi)
    cov_chi = cal_cov(var_chi)
    cov_t   = cal_cov(var_t)

    eig_value_z,   eig_vec_z   = np.linalg.eig(cov_z)
    eig_value_q,   eig_vec_q   = np.linalg.eig(cov_q)
    eig_value_psi, eig_vec_psi = np.linalg.eig(cov_psi)
    eig_value_chi, eig_vec_chi = np.linalg.eig(cov_chi)
    eig_value_t,   eig_vec_t   = np.linalg.eig(cov_t)

    eig_value = np.stack([eig_value_z, eig_value_q, eig_value_psi, eig_value_chi, eig_value_t]) # 5 x 13
    eig_vec   = np.stack([eig_vec_z, eig_vec_q, eig_vec_psi, eig_vec_chi, eig_vec_t])           # 5 x 13 x 13

    err_sample_vmode = np.copy(err_sample_sfvp)

    for i in range(5):
        sample = err_sample_sfvp[:, 4+13*i:4+13*(i+1), :, :].transpose(1, 0, 2, 3).reshape(13, -1)
        sample = np.matmul(np.sqrt(np.diag(eig_value[i]**(-1))), np.matmul(eig_vec[i].T, sample)).reshape(13, -1, lat, lon).transpose(1, 0, 2, 3)
        err_sample_vmode[:, 4+13*i:4+13*(i+1), :, :] = sample
        
    err_sample_vmode[:, 0:4] = err_sample_sfvp[:, 0:4] / std_sur.reshape(1, -1, 1, 1)
    b_matrix["vert_eig_vec"] = eig_vec
    b_matrix["vert_eig_value"] = eig_value
    b_matrix["std_sur"] = std_sur

    error_psi = err_sample_vmode[:, 4+13*2:4+13*3] # 240 x 13 x 128 x 256
    error_psi_arr = error_psi.transpose(1, 0, 2, 3).reshape(13, -1)
    # cov_psi   = np.cov(error_psi.transpose(1, 0, 2, 3).reshape(13, -1)) # 13 x 13
    cov_psi   = np.matmul(error_psi_arr, error_psi_arr.T) / error_psi_arr.shape[1]
    cov_psi_inv = np.linalg.inv(cov_psi)

    var_coeff = [0, 1, 2, 3]
    for i in [0, 3, 4]:
        for j in range(13):
            var_coeff.append(4 + i*13 + j)

    reg_coeff_list = np.zeros((69, 13))
    for layer in range(69):
        if layer in var_coeff:
            b_coeff = np.mean(err_sample_vmode[:, layer:layer+1] * error_psi, (0, 2, 3))
            reg_coeff_list[layer] = np.matmul(cov_psi_inv, b_coeff)
        else:
            reg_coeff_list[layer] = 0

    err_sample_static = np.copy(err_sample_vmode)

    for i in range(69):
        err_sample_static[:, i] = err_sample_vmode[:, i] - np.sum(error_psi * reg_coeff_list[i].reshape(1, -1, 1, 1), 1)

    b_matrix["reg_coeff"] = reg_coeff_list

    sph = Spharmt(err_sample_static.shape[3], err_sample_static.shape[2], gridtype='regular', rsphere=6.3712e6, legfunc="stored")

    len_scale_sample = np.zeros((err_sample_static.shape[0], err_sample_static.shape[1]))

    for i in range(err_sample_static.shape[0]):
        spec = sph.grdtospec(err_sample_static[i, :].transpose(1, 2, 0), ntrunc=None)
        dx, dy = sph.getgrad(spec)
        spec_dx = sph.grdtospec(dx, ntrunc=None)
        dxdx, dxdy1 = sph.getgrad(spec_dx)
        spec_dy = sph.grdtospec(dy, ntrunc=None)
        dxdy2, dydy = sph.getgrad(spec_dy)
        lap = np.transpose(dxdx + dydy, (2, 0, 1))
        len_scale_sample[i] = (8*np.var(err_sample_static[i], (1, 2))/np.var(lap, (1, 2)))**0.25

    len_scale = len_scale_sample.mean(0) / (111195 * 180 / lat)

    b_matrix["len_scale"] = len_scale

    return b_matrix
