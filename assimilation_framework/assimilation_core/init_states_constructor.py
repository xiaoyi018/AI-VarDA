import os
import io

import numpy as np

from assimilation_framework.utils.misc import load_config

class InitStatesConstructor:
    def __init__(self, config, client, save_path, forecast_model, field_reader, tstamp, dtime):
        self.type = config["type"]
        self.config = load_config(config["config"])
        self.forecast_model = forecast_model
        self.field_reader = field_reader
        self.tstamp = tstamp
        self.dtime  = dtime

        self.client = client
        self.save_path = save_path

    def construct_init_states(self):
        if self.type == "time_lagging":
            self.gap = self.config["gap"]
            field = self.field_reader.get_state(self.tstamp - self.gap * self.dtime)
            bg_field = self.forecast_model.do_forecast(field, self.gap)
            with io.BytesIO() as f:
                np.save(f, bg_field.numpy())
                f.seek(0)
                self.client.put(f"{self.save_path}/{self.tstamp.strftime('%Y-%m-%dT%H-%M')}/forecast_1.npy", f)

