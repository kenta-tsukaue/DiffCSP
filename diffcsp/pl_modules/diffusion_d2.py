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

from diffcsp.pl_modules.diff_utils import calculate_del1_delc, calculate_del1_delm, calculate_del2_delc, calculate_del2_delm, calculate_delI_delc_delc_delx_t, calculate_delI_delm_delm_delx_t, calculate_dellogp_delx_t, calculate_delx_t, d_log_p_wrapped_normal, d2_log_p_wrapped_normal, calculate_derivatives,  calculate_y_squared, check_sol_2, generate_tables, find_best_fit, calculate_I
from diffcsp.pl_modules.chksol_1 import check_sol
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
        self.keep_coords = self.hparams.cost_coord < 1e-5,

    def forward(self, batch):
        
        print(batch)
        #print("================[batch.y]===============\n",batch.y)
        #print("================[batch.frac_coords]===============\n",batch.frac_coords)
        #print("================[batch.atom_types]===============\n",batch.atom_types)
        #
        #     print(batch.atom_types[i])
        #print("================[batch.lengths]===============\n",batch.lengths)
        #print("================[batch.angles]===============\n",batch.angles)
        #print("================[batch.to_jimages]===============\n",batch.to_jimages)
        #print("================[batch.num_atoms]===============\n",batch.num_atoms)
        #print("================[batch.num_bonds]===============\n",batch.num_bonds)
        #print("================[batch.num_nodes]===============\n",batch.num_nodes)
        
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        print(times.shape)

        time_emb = self.time_embedding(times)
        print("time_emb", time_emb.shape)

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


        """
        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices
        """
        
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
    
    def check_d2(self, batch):
        time = 200
        print("タイムステップ:", time)
        batch_size = batch.num_graphs
        times = torch.full((batch_size,), time, device=self.device) #800に固定

        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        print("sigmas : ",sigmas)
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]
        

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        # Compute derivatives
        epsilon = 1e-2
        input_frac_coords = input_frac_coords.detach().clone()
        derivatives = torch.zeros_like(input_frac_coords)

        for i in range(input_frac_coords.shape[0]):
            for j in range(input_frac_coords.shape[1]):
                input_frac_coords_plus = input_frac_coords.clone()
                input_frac_coords_plus[i, j] += epsilon
                input_frac_coords_plus = input_frac_coords_plus
                
                input_frac_coords_minus = input_frac_coords.clone()
                input_frac_coords_minus[i, j] -= epsilon
                input_frac_coords_minus = input_frac_coords_minus

                _, pred_x_plus = self.decoder(time_emb, batch.atom_types, input_frac_coords_plus, input_lattice, batch.num_atoms, batch.batch)
                _, pred_x_minus = self.decoder(time_emb, batch.atom_types, input_frac_coords_minus, input_lattice, batch.num_atoms, batch.batch)

                derivative = (pred_x_plus - pred_x_minus) / (2 * epsilon)
                derivatives[i, j] = derivative.mean()

        
        print("x_0からx_tへの変化(x_t - x_0)\n", input_frac_coords.shape, "\n", sigmas_per_atom * rand_x)
        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        _, pred_x_d2 = self.decoder_d2(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
        
        print("予測されたスコア\n", pred_x.shape, "\n", pred_x)
        print("真のスコア\n", tar_x.shape, "\n", tar_x)
        tar_x_d2 = d2_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom) + tar_x**2

        print("予測されたスコアから求めた二次スコアDerivatives:\n", derivatives)
        print("予測された二次スコア:\n", pred_x_d2.shape,"\n", pred_x_d2)
        print("真のニ次スコア:\n", tar_x_d2.shape, "\n", d2_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom))
    
    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        """if self.keep_coords:
            print("keep_coords")
            x_T = batch.frac_coords

        if self.keep_lattice:
            print("keep_lattice")
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)"""

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
            _, pred_x_d2 = self.decoder_d2(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            #新しい方法でmとcを求める
            s1_table, s2_table, m_values, c_values = generate_tables(x_t)
            m, c = find_best_fit(s1_table, s2_table, m_values, c_values, pred_x, pred_x_d2, sigma_x)
            #print("m", m.shape, m)
            #print("c", c.shape, c)
        
            # yを求める
            y = calculate_y_squared(batch.num_atoms, batch)
            I = calculate_I(batch.num_atoms, batch, m, c)
            #print("y",y.shape, y)
            #print("I", I.shape, I)

            # ∂(1)/∂m, ∂(1)/∂cを求める
            del1_delm = calculate_del1_delm(m, c, sigma_x, x_t)
            del1_delc = calculate_del1_delc(m, c, sigma_x, x_t)

            #print("del1_delm", del1_delm.shape, del1_delm)
            #print("del1_delc", del1_delc.shape, del1_delc)

            # ∂(2)/∂m, ∂(2)/∂cを求める
            del2_delm = calculate_del2_delm(m, c, sigma_x, x_t)
            del2_delc = calculate_del2_delc(m, c, sigma_x, x_t)

            #print("del2_delm", del2_delm.shape, del2_delm)
            #print("del2_delc", del2_delc.shape, del2_delc)


            # ∂m/∂x_t, ∂c/x_tを求める   
            delm_delx, delc_delx = calculate_delx_t(del1_delm, del2_delm, del1_delc, del2_delc, pred_x, pred_x_d2)

            #print(delm_delx)
            #print(delc_delx)
            #print("delm_delx", delm_delx.shape, delm_delx)
            #print("delc_delx", delc_delx.shape, delc_delx)

            # ∂I/∂x_tを求める
            delI_delm_delm_delx_t = calculate_delI_delm_delm_delx_t(batch.num_atoms, batch, m, c, delm_delx)
            delI_delc_delc_delx_t = calculate_delI_delc_delc_delx_t(batch.num_atoms, batch, m, c, delc_delx)

            print("delI_delm_delm_delx_t", delI_delm_delm_delx_t.shape, delI_delm_delm_delx_t[0][0][0][0])
            print("delI_delc_delc_delx_t", delI_delc_delc_delx_t.shape, delI_delc_delc_delx_t[0][0][0][0])

            dellogp_delx_t = calculate_dellogp_delx_t(I, y, delI_delm_delm_delx_t, batch.num_atoms, sigma=0.5)

            print("dellogp_delx_t", dellogp_delx_t.shape, dellogp_delx_t)


            #dy = torch.tensor(dy).to('cuda').type(pred_x.dtype)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            #x_t_minus_05 = x_t - step_size * ( pred_x + dy ) + std_x * rand_x
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
            x_t_minus_05 = x_t_minus_05

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
    

    
    
    
    