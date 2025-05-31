import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from assimilation_framework.networks.fengwu_lr.transformer import LGUnet_all

class VAE_lr(nn.Module):
    def __init__(self, param_path, nlat, nlon, lora_rank=0):
        super(VAE_lr, self).__init__()
    
        with open(param_path, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

        self.param_encoder = cfg_params["encoder"]
        self.param_decoder = cfg_params["decoder"]

        self.param_encoder["rank"] = lora_rank
        self.param_decoder["rank"] = lora_rank
        
        self.enc = LGUnet_all(**self.param_encoder)
        self.dec = LGUnet_all(**self.param_decoder)
        self.nlat = nlat 
        self.nlon = nlon
        
    def encoder(self, x):
        encoded = self.enc(x)

        return encoded.chunk(2, dim = 1)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        # return cp.checkpoint(self.dec, z)
        return self.dec(z)
   
    def decoder_hr(self, z):
        # return cp.checkpoint(self.dec, z)
        x = self.dec(z)
        return F.interpolate(x, (self.nlat, self.nlon))

    def finetune(self,):
        for name, param in self.named_parameters():
            if name.split(".")[-2] in ["kA", "kB", "qA", "qB", "vA", "vB"]:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def loss_function(recon_x, x, mu, log_var, sigma):
    MSE = torch.sum((recon_x - x)**2 ) / (2 * sigma**2) #* (32 / imsize)**2
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD, MSE*2 * sigma**2, KLD
