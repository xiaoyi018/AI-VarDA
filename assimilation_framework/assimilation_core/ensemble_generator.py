import os
import io

import numpy as np
import torch

class EnsembleGenerator:
    def __init__(self, config, client, load_path, forecast_model):
        self.dynamic_mem = config["dynamic_member"]
        self.history_mem = config["history_member"]
        self.client = client
        self.load_path = load_path
        self.forecast_model = forecast_model

    def load_data_from_folder(self, folder):
        file_path = os.path.join(folder, "forecast_1.npy")
        with io.BytesIO(self.client.get(file_path)) as f:
            bg = torch.from_numpy(np.load(f)).to(torch.float32)

        all_data = [bg]

        for file in self.client.list(folder):
            file_path = os.path.join(folder, file)
            if file.endswith('.npy') and not file == "forecast_1.npy":
                with io.BytesIO(self.client.get(file_path)) as f:
                    data = torch.from_numpy(np.load(f)).to(torch.float32)
                all_data.append(data)
            else:
                continue 

        return all_data, bg

    def get_bg_ensemble(self, tstamp):
        if self.history_mem > 0:
            raise NotImplementedError
        
        folder = f"{self.load_path}/{tstamp.strftime('%Y-%m-%dT%H-%M')}"
        return self.load_data_from_folder(folder)

    def gen_bg_ensemble(self, xa, tstamp, dtime):
        field = xa
        current_time = tstamp
        for i in range(self.dynamic_mem):
            field = self.forecast_model.do_forecast(field, 1)
            current_time += dtime
            with io.BytesIO() as f:
                np.save(f, field.numpy())
                f.seek(0)
                self.client.put(f"{self.load_path}/{current_time.strftime('%Y-%m-%dT%H-%M')}/forecast_{i+1}.npy", f)

