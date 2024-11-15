import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import add_noise_to_structure, d_log_p_wrapped_normal, generate_crystal_structures, calculate_dellogp_delx_t_with_all_flow, calculate_loss, calculate_loss_unit
from diffcsp.pl_modules.crystal_utils import get_fq

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
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = False #self.hparams.cost_lattice < 1e-5
        self.keep_coords = False #self.hparams.cost_coord < 1e-5

    def forward(self, batch):

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
            #print("coords固定")
            input_frac_coords = frac_coords

        if self.keep_lattice:
            #print("lattice固定")
            input_lattice = lattices

        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        #print(pred_x)


        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)


        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):
        batch_loss_list = []
        loss_list = []
        batch_size = batch.num_graphs
        fq = get_fq(batch)

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

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)
            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
        
            x_t_minus_1 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

            c = torch.full((batch_size * len(pred_x), 3), 0.01) #5は適宜変更

            if t % 10 == 0 or t == 0:
                batch_loss = calculate_loss(batch, traj[t - 1], c, fq)
                loss = calculate_loss_unit(batch, traj[t - 1], c, fq)
                print("回折強度損失:", batch_loss)
                batch_loss_list.append(batch_loss)
                loss_list.append(loss)

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack, batch_loss_list, loss_list
    
    @torch.no_grad()
    def sample_new_method(self, batch, step_lr = 1e-5):
        # シード値を設定
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        batch_loss_list = []
        loss_list = []
        batch_size = batch.num_graphs
        fq = get_fq(batch)
        # print(fq.shape,fq)

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        print(x_T[0])

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

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            if t < 500:
                pred_x_d2 = torch.full(pred_x.shape, -1/sigma_x, device=pred_x.device) # -1/sigmaを使用
                
                dellogp_delx_t, m, c = calculate_dellogp_delx_t_with_all_flow(x_t, pred_x, pred_x_d2, sigma_x, batch, fq)

                x_t_minus_1 = x_t - step_size * (pred_x + 3 * dellogp_delx_t) + std_x * rand_x if not self.keep_coords else x_t
                #x_t_minus_1 = x_t - step_size * (0.2 * pred_x + 0.8 * dellogp_delx_t) + std_x * rand_x if not self.keep_coords else x_t
                #x_t_minus_1 = x_t - step_size * dellogp_delx_t + std_x * rand_x if not self.keep_coords else x_t
                # x_t_minus_1 = x_t - step_size * (pred_x - dellogp_delx_t) + std_x * rand_x if not self.keep_coords else x_t
            else:
                m = x_t
                c = torch.full(pred_x.shape, 0.01, device=pred_x.device)
                x_t_minus_1 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            
            l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'm': m              
            }
            if t % 10 == 0 or t == 0:
                batch_loss = calculate_loss(batch, traj[t - 1], c, fq)
                loss = calculate_loss_unit(batch, traj[t - 1], c, fq)
                print("回折強度損失:", batch_loss)
                batch_loss_list.append(batch_loss)
                loss_list.append(loss)
                
            if t % 100 == 0 or t == 0:
                torch.save(traj[t - 1], f'traj_{t}.pt')
                print("保存されました")

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }
        

        return traj[0], traj_stack, batch_loss_list, loss_list
    

    def check_negative_values(self, tensor, name="tensor"):
        if (tensor < 0).any():
            print(f"Warning: Negative values found in {name}")

    def save_hist(self, data,idx, t, tag):
        # (8, 3)の部分テンソルを取得
        subset = data[:8]  # 最初の8行を取得
        flattened_data = subset.view(-1).cpu().numpy()  # 1次元に変換（24要素）

        # ヒストグラムをプロットして保存
        # Plotlyでヒストグラムを作成
        fig = px.histogram(flattened_data, nbins=100, title="Histogram of First 8 (n,3) Tensor Elements")
        fig.update_layout(xaxis_title="Value", yaxis_title="Frequency")

        # ファイル名を設定して保存
        filename = f"{tag}_{idx}_{t}.png"
        pio.write_image(fig, filename)
        print(f"ヒストグラムを保存しました: {filename}")

    
    @torch.no_grad()
    def sample_new_method_and_random(self, idx, batch, step_lr = 1e-5):
        print(batch)
        # シード値を設定
        torch.manual_seed(idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(idx)

        batch_loss_list = []
        loss_list = []

        batch_loss_list_new = []
        loss_list_new = []
        fig_list = []

        batch_size = batch.num_graphs
        fq = get_fq(batch)
        # print(fq.shape,fq)

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        print(x_T[0]) #tensor([0.8107, 0.2416, 0.4292], device='cuda:0')

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

        traj_new = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        # factor の候補値を定義
        #factor_candidates = [3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
        factor_candidates = [3]
        current_factor = 3 # 初期値

        # 次の factor 更新ステップ
        next_update_step = time_start


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
            self.check_negative_values(x_t, name="x_t")
            l_t = traj[t]['lattices']

            x_t_new = traj_new[t]['frac_coords']
            self.check_negative_values(x_t_new, name="x_t_new")
            l_t_new = traj_new[t]['lattices']

            if self.keep_coords:
                x_t = x_T
                x_t_new = x_T

            if self.keep_lattice:
                l_t = l_T
                l_t_new = l_T

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            pred_l_new, pred_x_new = self.decoder(time_emb, batch.atom_types, x_t_new, l_t_new, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            pred_x_new = pred_x_new * torch.sqrt(sigma_norm)

            if t <= 1000:
                pred_x_d2 = torch.full(pred_x_new.shape, -1/sigma_x, device=pred_x_new.device) # -1/sigmaを使用
                
                dellogp_delx_t, dellogp_delx_t_pre, m, c = calculate_dellogp_delx_t_with_all_flow(x_t_new, pred_x_new, pred_x_d2, sigma_x, batch, fq)

                # factor を10ステップごとに更新
                if t == next_update_step:
                    best_loss = float('inf')
                    best_factor = current_factor

                    for factor in factor_candidates:
                        if t < 1000:
                            # 新しい factor を適用
                            x_t_minus_1_new_candidate = x_t_new - step_size * (pred_x_new + factor * dellogp_delx_t) if not self.keep_coords else x_t_new

                            # 仮のトラジェクトリを作成
                            traj_new_candidate = traj_new.copy()
                            traj_new_candidate[t - 1] = {
                                'num_atoms': batch.num_atoms,
                                'atom_types': batch.atom_types,
                                'frac_coords': x_t_minus_1_new_candidate % 1.,
                                'lattices': l_t_minus_1_new,
                                'm': m
                            }

                            # ロスを計算
                            batch_loss_new_candidate = calculate_loss(batch, traj_new_candidate[t - 1], c, fq)

                            if batch_loss_new_candidate < best_loss:
                                best_loss = batch_loss_new_candidate
                                best_factor = factor

                    current_factor = best_factor
                    print(f"Step {t}: Selected best factor {current_factor} with loss {best_loss}")
                    next_update_step = t - 10  # 次の更新ステップを設定

                x_t_minus_1_new = x_t_new - step_size * (pred_x_new + current_factor * dellogp_delx_t) if not self.keep_coords else x_t_new


            else:
                m = x_t_new
                c = torch.full(pred_x.shape, 0.01, device=pred_x.device)
                #x_t_minus_1_new = x_t_new - step_size * pred_x_new + std_x * rand_x if not self.keep_coords else x_t_new
                x_t_minus_1_new = x_t_new - step_size * pred_x_new if not self.keep_coords else x_t_new

            #x_t_minus_1 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            x_t_minus_1 = x_t - step_size * pred_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            l_t_minus_1_new = c0 * (l_t_new - c1 * pred_l_new) + sigmas * rand_l if not self.keep_lattice else l_t_new

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'm': m              
            }

            traj_new[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1_new % 1.,
                'lattices' : l_t_minus_1_new,
                'm': m            
            }
            if t % 10 == 0 or t == 0:
                batch_loss = calculate_loss(batch, traj[t - 1], c, fq)
                batch_loss_new = calculate_loss(batch, traj_new[t - 1], c, fq)
                loss = calculate_loss_unit(batch, traj[t - 1], c, fq)
                loss_new = calculate_loss_unit(batch, traj_new[t - 1], c, fq)
                print("回折強度損失(ランダム):", batch_loss)
                print("回折強度損失(新手法):", batch_loss_new)
                batch_loss_list.append(batch_loss)
                loss_list.append(loss)
                batch_loss_list_new.append(batch_loss_new)
                loss_list_new.append(loss_new)

        return traj[0], traj_new[0], batch_loss_list, loss_list, batch_loss_list_new, loss_list_new
    
    def repeat_20(self, batch):
        # 元のバッチのデータを取得
        edge_index = batch.edge_index
        y = batch.y
        frac_coords = batch.frac_coords
        atom_types = batch.atom_types
        lengths = batch.lengths
        angles = batch.angles
        to_jimages = batch.to_jimages
        num_atoms = batch.num_atoms
        num_bonds = batch.num_bonds
        num_nodes = batch.num_nodes
        batch_indices = batch.batch
        ptr = batch.ptr

        # 各フィールドを20倍にする
        edge_index_20 = edge_index.repeat(1, 20)
        y_20 = y.repeat(20, 1)
        frac_coords_20 = frac_coords.repeat(20, 1)
        atom_types_20 = atom_types.repeat(20)
        lengths_20 = lengths.repeat(20, 1)
        angles_20 = angles.repeat(20, 1)
        to_jimages_20 = to_jimages.repeat(20, 1)
        num_atoms_20 = num_atoms.repeat(20)
        num_bonds_20 = num_bonds.repeat(20)
        num_nodes_20 = num_nodes * 20
        batch_indices_20 = batch_indices.repeat(20)
        ptr_20 = ptr.repeat(20)

        # 新しいデータバッチを作成
        new_batch = Data(
            edge_index=edge_index_20,
            y=y_20,
            frac_coords=frac_coords_20,
            atom_types=atom_types_20,
            lengths=lengths_20,
            angles=angles_20,
            to_jimages=to_jimages_20,
            num_atoms=num_atoms_20,
            num_bonds=num_bonds_20,
            num_nodes=num_nodes_20,
            batch=batch_indices_20,
            ptr=ptr_20
        )

        return new_batch

    

    @torch.no_grad()
    def sample_new_method_and_random_20(self, idx, batch, step_lr = 1e-5):
        print(batch)
        if batch.num_graphs != 1:
            print("バッチサイズを1にしてください")
            exit()
        batch = self.repeat_20(batch)

        batch_loss_list = []
        loss_list = []

        batch_loss_list_new = []
        loss_list_new = []

        batch_size = 20
        fq = get_fq(batch)
        # print(fq.shape,fq)

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

        traj_new = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        # factor の候補値を定義
        #factor_candidates = [3]
        factor_candidates = [3]
        current_factor = 3 # 初期値

        # 次の factor 更新ステップ
        next_update_step = time_start


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
            self.check_negative_values(x_t, name="x_t")
            l_t = traj[t]['lattices']

            x_t_new = traj_new[t]['frac_coords']
            self.check_negative_values(x_t_new, name="x_t_new")
            l_t_new = traj_new[t]['lattices']

            if self.keep_coords:
                x_t = x_T
                x_t_new = x_T

            if self.keep_lattice:
                l_t = l_T
                l_t_new = l_T

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            pred_l_new, pred_x_new = self.decoder(time_emb, batch.atom_types, x_t_new, l_t_new, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            pred_x_new = pred_x_new * torch.sqrt(sigma_norm)

            if t <= 1000:
                pred_x_d2 = torch.full(pred_x_new.shape, -1/sigma_x, device=pred_x_new.device) # -1/sigmaを使用
                
                dellogp_delx_t, dellogp_delx_t_pre, m, c = calculate_dellogp_delx_t_with_all_flow(x_t_new, pred_x_new, pred_x_d2, sigma_x, batch, fq)

                # factor を10ステップごとに更新
                if t == next_update_step:
                    best_loss = float('inf')
                    best_factor = current_factor

                    for factor in factor_candidates:
                        if t < 1000:
                            # 新しい factor を適用
                            x_t_minus_1_new_candidate = x_t_new - step_size * (pred_x_new + factor * dellogp_delx_t) + std_x * rand_x if not self.keep_coords else x_t_new

                            # 仮のトラジェクトリを作成
                            traj_new_candidate = traj_new.copy()
                            traj_new_candidate[t - 1] = {
                                'num_atoms': batch.num_atoms,
                                'atom_types': batch.atom_types,
                                'frac_coords': x_t_minus_1_new_candidate % 1.,
                                'lattices': l_t_minus_1_new,
                                'm': m
                            }

                            # ロスを計算
                            batch_loss_new_candidate = calculate_loss(batch, traj_new_candidate[t - 1], c, fq)

                            if batch_loss_new_candidate < best_loss:
                                best_loss = batch_loss_new_candidate
                                best_factor = factor

                    current_factor = best_factor
                    #print(f"Step {t}: Selected best factor {current_factor} with loss {best_loss}")
                    next_update_step = t - 10  # 次の更新ステップを設定

                x_t_minus_1_new = x_t_new - step_size * (pred_x_new + current_factor * dellogp_delx_t) + std_x * rand_x if not self.keep_coords else x_t_new


            else:
                m = x_t_new
                c = torch.full(pred_x.shape, 0.01, device=pred_x.device)
                x_t_minus_1_new = x_t_new - step_size * pred_x_new + std_x * rand_x if not self.keep_coords else x_t_new
                #x_t_minus_1_new = x_t_new - step_size * pred_x_new if not self.keep_coords else x_t_new

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            #x_t_minus_1 = x_t - step_size * pred_x if not self.keep_coords else x_t

            l_t_minus_05 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            l_t_minus_1_new = c0 * (l_t_new - c1 * pred_l_new) + sigmas * rand_l if not self.keep_lattice else l_t_new

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'm': m              
            }

            traj_new[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1_new % 1.,
                'lattices' : l_t_minus_1_new,
                'm': m            
            }
            if t % 10 == 0 or t == 0:
                batch_loss = calculate_loss(batch, traj[t - 1], c, fq)
                batch_loss_new = calculate_loss(batch, traj_new[t - 1], c, fq)
                loss = calculate_loss_unit(batch, traj[t - 1], c, fq)
                loss_new = calculate_loss_unit(batch, traj_new[t - 1], c, fq)
                print("回折強度損失(ランダム):", batch_loss)
                print("回折強度損失(新手法):", batch_loss_new)
                batch_loss_list.append(batch_loss)
                loss_list.append(loss)
                batch_loss_list_new.append(batch_loss_new)
                loss_list_new.append(loss_new)

        return batch, traj[0], traj_new[0], batch_loss_list, loss_list, batch_loss_list_new, loss_list_new
    

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord},
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
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord
        }

        return log_dict, loss
    

    