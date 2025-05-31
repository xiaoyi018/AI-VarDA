import json
import functools
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .base_decoder import BackgroundDecoder
from assimilation_framework.networks.bg_vae import VAE_lr

class VAEDecoder(BackgroundDecoder):
    """
    Decoder network from a trained VAE model.
    Maps latent control variable z to background field x.
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
        self.ckpt_path = config["checkpoint"]
        self.lora_rank = config.get("lora_rank", 0)
        self.build_vae_model(config["param_path"])
        with open(config["model_mean_std"], 'r') as file:
            self.model_std = torch.from_numpy(np.array(json.load(file)["std"])).float().to(self.device)
        with open(config["err_std"], 'r') as file:
            self.err_std = torch.from_numpy(np.array(json.load(file)["std"])).float().to(self.device)

    def build_vae_model(self, param_path):
        self.vae_model = VAE_lr(param_path, self.nlat_out, self.nlon_out, self.lora_rank).to(self.device)
        self.load_checkpoint(self.ckpt_path)

    def load_checkpoint(self, ckpt_path):
        checkpoint_model = torch.load(ckpt_path, map_location='cuda:0')
        new_state_dict = self.vae_model.state_dict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        self.vae_model.load_state_dict(new_state_dict)
        self.vae_model.eval()

    def save_checkpoint(self, ckpt_path):
        torch.save(self.vae_model.state_dict(), ckpt_path)

    def save_lora_checkpoint(self, ckpt_path):
        filtered_state_dict = {
            name: param
            for name, param in self.vae_model.state_dict().items()
            if name.split(".")[-2] in ["kA", "kB", "qA", "qB", "vA", "vB"]
        }
        torch.save(filtered_state_dict, ckpt_path)

    def decode(self, z):
        z = z.unsqueeze(0)
        x = self.vae_model.decoder_hr(z).squeeze(0)
        x = (x * self.err_std.reshape(-1, 1, 1)) * self.model_std.reshape(-1, 1, 1)
        return x

    def process_data(self, sample):
        data = torch.stack(sample, 0).numpy()
        data_mean = np.mean(data, axis=0)
        data -= data_mean
        data_std  = data.std((2, 3))
        data_norm = data / data_std.reshape(*data_std.shape, 1, 1)
        data_std_sample = data_norm.std(0).mean((1, 2))
        data_processed = data_norm / data_std_sample.reshape(1, -1, 1, 1)
        return data_processed

    def _train_sample(self, optimizer, sample, lr, epoch, bs):
        processed_data = self.process_data(sample)
        data_tensor = torch.tensor(processed_data, dtype=torch.float32)

        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        for i in range(epoch):
            total_loss = 0
            total_rec_loss = 0
            total_kld_loss = 0
            total_num = 0
            for j, err in enumerate(dataloader):
                err = err[0].to(self.device)
                err = F.interpolate(err, (128, 256))
                recon_batch, mu, log_var = self.vae_model(err)
                loss, rec_loss, kld_loss = self.loss_function(recon_batch, err, mu, log_var, 2.00)
                total_loss += loss
                total_rec_loss += rec_loss
                total_kld_loss += kld_loss
                total_num += err.shape[0]
                self.vae_model.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss     = total_loss / total_num
            avg_rec_loss = total_rec_loss / total_num
            avg_kld_loss = total_kld_loss / total_num
            self.logger.info(f"[Finetuing epoch: {i}] loss: {avg_loss:.3f} rec loss: {avg_rec_loss:.3f} kld loss: {avg_kld_loss:.3f}")

    def trainer(self, sample, lr, epoch, bs, finetune_type):
        if finetune_type == "full_finetune":
            self.vae_model.train()
        elif finetune_type == "reset_finetune":
            self.vae_model = self.load_checkpoint(self.vae_model, self.ckpt_path)
            self.vae_model.train()
        elif finetune_type == "lora_finetune":
            self.vae_model.finetune()

        optimizer = torch.optim.Adam([param for param in self.vae_model.parameters() if param.requires_grad], lr=lr)
        total_params = sum(p.numel() for p in self.vae_model.parameters() if p.requires_grad)
        self.logger.info(f"Total trainable parameters: {total_params}")

        @self.timeit(finetune_type)
        def train_sample(): 
            self._train_sample(optimizer, sample, lr, epoch, bs)

        train_sample()

    def loss_function(self, recon_x, x, mu, log_var, sigma):
        MSE = torch.sum((recon_x - x)**2 ) / (2 * sigma**2) #* (32 / imsize)**2
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD, MSE*2 * sigma**2, KLD

    def timeit(self, label):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                if self.logger:
                    self.logger.info(f"{label} finished. Time consumed: {end - start:.2f} (s)")
                return result
            return wrapper
        return decorator
