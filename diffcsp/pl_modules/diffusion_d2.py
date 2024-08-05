import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal, d2_log_p_wrapped_normal, optimize_mc

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optim.optimizer,params=self.parameters(), _convert_="partial")  # YAMLからオプティマイザ設定を読み込み

        if self.hparams.optim.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(self.hparams.optim.lr_scheduler, optimizer=optimizer)
            scheduler_config = {
                'scheduler': scheduler,
                'monitor': 'val_loss', 
                'interval': 'epoch',
                'frequency': 100
            }
            return [optimizer], [scheduler_config]
        else:
            return optimizer


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.decoder_d2 = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5

    def forward(self, batch):
        """
        print(batch)
        print("================[batch.y]===============\n",batch.y)
        print("================[batch.frac_coords]===============\n",batch.frac_coords)
        print("================[batch.atom_types]===============\n",batch.atom_types)
        for i in range(len(batch.atom_types)):
            print(batch.atom_types[i])
        print("================[batch.lengths]===============\n",batch.lengths)
        print("================[batch.angles]===============\n",batch.angles)
        print("================[batch.to_jimages]===============\n",batch.to_jimages)
        print("================[batch.num_atoms]===============\n",batch.num_atoms)
        print("================[batch.num_bonds]===============\n",batch.num_bonds)
        print("================[batch.num_nodes]===============\n",batch.num_nodes)
        """
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)

        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]
        

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        _, pred_x_d2 = self.decoder_d2(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        #print("==============[tar_x]==============\n", tar_x.size(),"\n", tar_x)
        tar_x_d2 = d2_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom) + tar_x**2
        #print("==============[tar_x_d2]==============\n", tar_x_d2.size(),"\n", tar_x_d2)



        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_coord_d2 = F.mse_loss(pred_x_d2 + pred_x**2, tar_x_d2)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_coord_d2 * loss_coord_d2)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_coord_d2' : loss_coord_d2
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            """if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T"""

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch) #スコア算出
            _, pred_x_d2 = self.decoder_d2(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch) #二次スコア算出

            #この2次スコアを用いてまずはmとcを求める
            #m, c = optimize_mc(x_t, sigma_x, pred_x, pred_x_d2 + pred_x ** 2)


            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            l_t_minus_05 = l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
            
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_coord_d2 = output_dict['loss_coord_d2']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'coord_d2_loss': loss_coord_d2},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)
        log_dict, val_loss = self.compute_stats(output_dict, prefix='val')

        print(log_dict)
        print(val_loss)

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return val_loss


    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        print(prefix)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_coord_d2 = output_dict['loss_coord_d2']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_coord_d2_loss': loss_coord_d2
        }

        return log_dict, loss
    

    