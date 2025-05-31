import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
import numpy as np

from assimilation_framework.utils.misc import load_config, DistributedParallel_Model, check_ddp_consistency
from assimilation_framework.networks.fengwu_lr.transformer import LGUnet_all
from assimilation_framework.networks.bg_vae import VAE_lr, loss_function
from assimilation_framework.bgerr_learning.utils import DatasetBuilder

class VaeBeAgent:
    def __init__(self, client, logger, args, config, run_dir):
        self.client  = client
        self.logger  = logger
        self.config  = config
        self.args    = args
        self.run_dir = run_dir
        self.vae_ckpt = os.path.join(run_dir, "checkpoints", "vae_ckpt")
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.epoch_path = os.path.join(run_dir, "current_epoch.txt")
        self.config_fengwu = load_config(config["fengwu_config"])
        self.config_fengwu['dataloader']['num_workers'] = args.per_cpus
        self.config_fengwu['dataset']['train']['length'] = config["length"]

        self.logger.info('Building models ...')
        self.vae_nmc_model = VaeNmcModel(self.logger, config["vae_parameter_group"], config["sigma"], config["fengwu_checkpoint"], self.vae_ckpt, config["device"], self.config_fengwu["model"], config["error_std_path"], self.epoch_path)

        self.model_without_ddp = DistributedParallel_Model(self.vae_nmc_model, args.local_rank)

        if args.world_size > 1:
            check_ddp_consistency(self.model_without_ddp.vae)
            check_ddp_consistency(self.model_without_ddp.fengwu)

        self.builder = DatasetBuilder(self.config_fengwu)

        params = [p for p in self.model_without_ddp.vae.parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])
        self.logger.info("params: {cnt_params}".format(cnt_params=cnt_params))

    def run(self):
        self.logger.info('Building dataloaders ...')
        train_dataloader = self.builder.get_dataloader(split='train', batch_size=self.config["batch_size"])
        self.logger.info('Dataloaders build complete')

        self.logger.info('Begin training VAE ...')
        self.vae_nmc_model.train(train_dataloader, self.config["epoch"], self.config["learning_rate"])
        self.logger.info('Training VAE complete')

class VaeNmcModel(nn.Module):
    def __init__(self, logger, param_str, sigma, path_fengwu, path_vae, device, params, err_std_path, epoch_path) -> None:
        super().__init__()

        self.params = params
        self.device = device
        self.logger = logger
        self.epoch_path = epoch_path

        self.ckpt_fengwu = path_fengwu
        self.ckpt_vae   = path_vae
        self.params = params.get("network_params", None)

        self.fengwu = LGUnet_all(**self.params).to(self.device)
        self.fengwu.eval()

        self.sigma = sigma
        self.param_str = param_str

        self.vae = VAE_lr(param_path=param_str, nlat=128, nlon=256).to(self.device)
        self.vae.train()

        with open(err_std_path, mode='r') as f:
            self.err_std = torch.Tensor(json.load(f)["std"]).to(self.device).reshape(1, 69, 1, 1)

        self.current_epoch = 0
        self.load_checkpoint()
        
    def load_checkpoint(self):
        checkpoint_dict = torch.load(self.ckpt_fengwu)
        checkpoint_model = checkpoint_dict['model']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        self.fengwu.load_state_dict(new_state_dict)

        if os.path.exists(self.epoch_path):
            checkpoint_dict = torch.load(self.ckpt_vae)
            new_state_dict = OrderedDict()
            for k, v in checkpoint_dict.items():
                if "module" == k[:6]:
                    name = k[7:]
                else:
                    name = k
                if not name == "max_logvar" and not name == "min_logvar":
                    new_state_dict[name] = v
            self.vae.load_state_dict(new_state_dict)
            f = open(self.epoch_path, "r")
            self.current_epoch = int(f.read())

    def train(self, data_loader, epoch_num=20, lr=1e-4):
        self.fengwu.eval()
        optimizer = optim.Adam(self.vae.parameters(), lr=lr)

        self.logger.info(str(len(data_loader)))
        self.vae.train()
        
        for i in range(self.current_epoch, epoch_num):
            
            for j, batch in enumerate(data_loader):

                batch = torch.stack(batch, 1).to(self.device)  # B x 5 x 69 x 128 x 256
                pred1 = batch[:, 0]
                for ii in range(8):
                    pred1 = self.fengwu(pred1)[:,:69].detach()
                pred2 = batch[:, 4]
                for ii in range(4):
                    pred2 = self.fengwu(pred2)[:,:69].detach()

                err = (pred2 - pred1) / self.err_std

                pred1 = pred1.cpu()
                pred2 = pred2.cpu()

                recon_batch, mu, log_var = self.vae(err)

                loss, rec_loss, kld_loss = loss_function(recon_batch, err, mu, log_var, self.sigma)

                self.vae.zero_grad()

                loss.backward()
                warmup_lr = lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()

                if (j+1) % 10 == 0 or (j+1) == len(data_loader):
                    self.logger.info("epoch: %d iter number: %d loss: %.3f rec loss: %.3f kld loss: %.3f"%(i, j, loss, rec_loss, kld_loss))

            self.logger.info("saving model")
            torch.save(self.vae.state_dict(), self.ckpt_vae)
            with open(self.epoch_path, 'w') as f:
                f.write(str(i+1))

            torch.cuda.empty_cache()
