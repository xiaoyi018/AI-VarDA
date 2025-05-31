import os
import io

import torch
import numpy as np

class FieldReader:
    def __init__(self, field_type, client):
        self.type     = field_type
        self.client   = client

    def get_state_from_era5_1440(self, tstamp, data_dir="cluster3:s3://era5_np_float32"):
        state = []
        single_level_vnames = ['u10', 'v10', 't2m', 'msl']
        multi_level_vnames = ['z','q', 'u', 'v', 't']
        height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        for vname in single_level_vnames:
            file = os.path.join('single/'+str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            url = f"{data_dir}/{file}-{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                state.append(np.load(f).reshape(1, 721, 1440))
        for vname in multi_level_vnames:
            file = os.path.join(str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            for idx in range(13):
                height = height_level[idx]
                url = f"{data_dir}/{file}-{vname}-{height}.0.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    state.append(np.load(f).reshape(1, 721, 1440))
        state = np.concatenate(state, 0)
        return torch.from_numpy(state).to(torch.float32)

    def get_state(self, tstamp):
        if self.type == "era5_1440":
            return self.get_state_from_era5_1440(tstamp)
        else:
            raise NotImplementedError
