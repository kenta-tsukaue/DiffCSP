from datetime import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.integrate import quad
import math
from torch.autograd import Variable
from scipy.optimize import minimize
import torch.optim as optim
import torch.nn.functional as F
from scipy.misc import derivative
from diffcsp.pl_modules.chksol_1 import loss_function_sol
from diffcsp.pl_modules.crystal_utils import complex_sum_squared_with_scattering_factors, calculate_q_magnitude, scattering_factor
from scipy.special import erf
from scipy.constants import pi
import torch.multiprocessing as mp



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N+1):
        p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)

    return p_ # p_.size() torch.Size([2362, 3])

def log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    # 総和の初期化
    sum_exp = 0
    # 各項を計算して総和を取る
    for i in range(-N, N+1):
        sum_exp += torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))

    # 対数を取る
    log_p = torch.log(sum_exp)

    return log_p

def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N+1):
        p_ += (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)

    return p_ / p_wrapped_normal(x, sigma, N, T) #size() torch.Size([2362, 3])

def d_p_wrapped_normal(x, sigma, N=10, T=1.0):
    dp_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        dp_ += (-(x + T * i) / (sigma ** 2)) * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return dp_

def d2_p_wrapped_normal(x, sigma, N=10, T=1.0):
    d2p_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        d2p_ += (((x + T * i)**2 / (sigma**4)) - (1 / (sigma**2))) * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return d2p_

def d2_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p = p_wrapped_normal(x, sigma, N, T)
    #print("p.size()",p.size())
    dp = d_p_wrapped_normal(x, sigma, N, T)
    #print("dp.size()",dp.size())
    d2p = d2_p_wrapped_normal(x, sigma, N, T)
    #print("d2p.size()", d2p.size())
    d2_log_p = (d2p * p - dp**2) / p**2

    """
    print("sigma", sigma.shape, sigma)
    print("x", x.shape, x)
    print("d2_log_p / torch.sqrt(sigma)", d2_log_p.size(), d2_log_p/torch.sqrt(sigma))
    """
    return d2_log_p

# target1 と target2 を計算する関数
def calculate_targets(decoder, decoder_d2, time_emb, atom_types, x_t, l_t, num_atoms, batch):
    pred_l, pred_x = decoder(time_emb, atom_types, x_t, l_t, num_atoms, batch)
    _, pred_x_d2 = decoder_d2(time_emb, atom_types, x_t, l_t, num_atoms, batch)
    target1 = pred_x
    target2 = pred_x_d2 + pred_x ** 2
    return target1, target2


# m, c の計算 & table作成
def calculate_s1(m, c, x_t):
    k = np.arange(-10, 11)[:, np.newaxis, np.newaxis]  # (21, 1, 1)
    xt_expanded = x_t[np.newaxis, :, :]  # (1, n, d)
    m_expanded = m - xt_expanded  # (1, n, d)

    erf_term1 = erf((-(1/2) + k + m_expanded) / np.sqrt(2 * c))
    erf_term2 = erf((1/2 + k + m_expanded) / np.sqrt(2 * c))
    s1_result = -np.sum(k / 2 * (erf_term1 - erf_term2), axis=0) + m_expanded
    return s1_result

def calculate_s2(m, c, x_t):
    k = np.arange(-10, 11)[:, np.newaxis, np.newaxis]  # (21, 1, 1)
    xt_expanded = x_t[np.newaxis, :, :]  # (1, n, d)
    m_expanded = m - xt_expanded  # (1, n, d)

    exp_term1 = np.exp(-((-(1/2) + k + m_expanded) ** 2 / (2 * c)))
    exp_term2 = np.exp(-((1/2 + k + m_expanded) ** 2 / (2 * c)))
    term1 = -np.sqrt(c / (2 * pi)) * np.sum((1/2 + k) * exp_term1 - (-(1/2) + k) * exp_term2, axis=0)
    
    erf_term1 = erf((-(1/2) + k + m_expanded) / np.sqrt(2 * c))
    erf_term2 = erf((1/2 + k + m_expanded) / np.sqrt(2 * c))
    term2 = -np.sum(1/2 * (k ** 2 + 2 * k * m_expanded) * (erf_term1 - erf_term2), axis=0)
    
    s2_result = term1 + term2 + c + m_expanded ** 2
    return s2_result

def generate_tables(x_t):
    x_t_cpu = x_t.cpu().detach().numpy()
    m_values = np.arange(-0.5, 0.5 + 1/20, 1/20)
    #c_values = np.arange(1e-2, 2 + 1/10, 1/10)
    #m_values = np.arange(-0.5, 0.5 + 1/20, 1/5000)
    c_values = np.arange(1e-2, 2e-2, 1.0)

    n, d = x_t_cpu.shape
    s1_table = np.zeros((n, d, len(m_values), len(c_values)))
    s2_table = np.zeros((n, d, len(m_values), len(c_values)))
    
    for i, m in enumerate(m_values):
        for j, c in enumerate(c_values):
            s1_results = calculate_s1(m, c, x_t_cpu)
            s2_results = calculate_s2(m, c, x_t_cpu)
            s1_table[:, :, i, j] = s1_results
            s2_table[:, :, i, j] = s2_results
    
    return s1_table, s2_table, m_values, c_values

def calculate_batch_error(batch, s1_table, s2_table, score1, score2, sigma2, sigma4, m_table, c_table):
    results = []
    for (i, j) in batch:
        s1 = s1_table[i, j, :, :]
        s2 = s2_table[i, j, :, :]
        
        err = (sigma2 * score1[i, j] - s1) ** 2 + (sigma4 * score2[i, j] + sigma2 - s2) ** 2
        min_idx = torch.argmin(err)
        kmin, lmin = divmod(min_idx.item(), s1.shape[1])
        
        results.append((i, j, m_table[kmin], c_table[lmin]))
    
    return results

def find_best_fit(s1_table, s2_table, m_table, c_table, score1, score2, sigma, num_workers=4):
    if not isinstance(s1_table, torch.Tensor):
        s1_table = torch.tensor(s1_table).to(sigma.device)
    if not isinstance(s2_table, torch.Tensor):
        s2_table = torch.tensor(s2_table).to(sigma.device)
    if not isinstance(m_table, torch.Tensor):
        m_table = torch.tensor(m_table).to(sigma.device)
    if not isinstance(c_table, torch.Tensor):
        c_table = torch.tensor(c_table).to(sigma.device)

    s1_table = s1_table.cpu()
    s2_table = s2_table.cpu()
    m_table = m_table.cpu()
    c_table = c_table.cpu()
    score1 = score1.cpu()
    score2 = score2.cpu()
    sigma2 = sigma.cpu() ** 2
    sigma4 = sigma.cpu() ** 4

    n, d, m_len, c_len = s1_table.shape
    
    m = torch.zeros((n, d), dtype=sigma.dtype, device=sigma.device)
    c = torch.zeros((n, d), dtype=sigma.dtype, device=sigma.device)

    # (i, j) の組み合わせをすべて列挙
    tasks = [(i, j) for i in range(n) for j in range(d)]
    
    # タスクをバッチに分割
    batch_size = max(1, len(tasks) // num_workers)
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(calculate_batch_error, [(batch, s1_table, s2_table, score1, score2, sigma2, sigma4, m_table, c_table) for batch in batches])
        pool.close()  # プールを閉じる
        pool.join()   # すべてのプロセスが終了するのを待つ
    
    # 結果を m と c に反映
    for batch_results in results:
        for i, j, m_val, c_val in batch_results:
            m[i, j] = m_val
            c[i, j] = c_val
    
    return m.cpu(), c.cpu()

# あるhklにおいての構造因子の値を出す
def I_hkl(k, A_m, A_c):
    #3次元ベクトル k と (n x 3) の行列 A_m, A_c を受け取り、
    #I(h,k,l) = sum_j sum_k exp{2 * pi * I [(m_j - m_k)・(h,k,l) - 2 * pi^2 (c_j + c_k)・(h^2, k^2, l^2)]}
    #を計算する関数。

    # Parameters:
    # k (np.ndarray): 3次元ベクトル (h, k, l)
    # A_m (np.ndarray): (n x 3) の行列 m_j
    # A_c (np.ndarray): (n x 3) の行列 c_j

    # Returns:
    # complex: 複素数の和
    i = complex(0, 1)
    pi = np.pi

    # CUDAテンソルをCPUに移動させてNumPy配列に変換
    if isinstance(A_m, torch.Tensor):
        A_m = A_m.cpu().numpy()
    if isinstance(A_c, torch.Tensor):
        A_c = A_c.cpu().numpy()

    # 行数を取得
    n = A_m.shape[0]

    # 複素数の和を計算
    result = 0.0
    for j in range(n):
        for m in range(n):
            diff_m = A_m[j] - A_m[m]
            sum_c = A_c[j] + A_c[m]
            r_m = np.dot(diff_m, k)
            r_c = np.dot(sum_c, k**2)
            result += np.exp(2 * pi * i * r_m - 2 * pi**2 * r_c)

    return result.real



# 予測されるm, cから構造因子を算出する
def calculate_I( batch, m, c):
    num_crystals = batch['num_atoms'].size(0)  # バッチサイズ
    # kの範囲設定
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)
    # 結果を格納する配列
    Z = np.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values)))

    # バッチ内の全ての結晶に対してループ
    for I in range(num_crystals):
        start_index = sum(batch['num_atoms'][:I])  # I番目の結晶の開始インデックス
        end_index = start_index + batch['num_atoms'][I]  # I番目の結晶の終了インデックス
        frac_coords_m = m[start_index:end_index]
        frac_coords_c = c[start_index:end_index]
        atom_types = batch["atom_types"][start_index:end_index]
        # k1とk2を動かしてcomplex_sumの値を計算
        for j, k1 in enumerate(k1_values):
            for l, k2 in enumerate(k2_values):
                for n, k3 in enumerate(k3_values):
                    k = np.array([k1, k2, k3])
                    # Z[I, j, l, n] = I_hkl(k, frac_coords_m, frac_coords_c)
                    Z[I, j, l, n] = complex_sum_squared_with_scattering_factors(k, frac_coords_m, frac_coords_c, atom_types)
    return Z

# 真の構造因子を算出
def calculate_y(batch, c):
    num_crystals = batch['num_atoms'].size(0)  # バッチサイズ
    # kの範囲設定
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)
    # 結果を格納する配列
    Z = np.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values)))

    # バッチ内の全ての結晶に対してループ
    for I in range(num_crystals):
        start_index = sum(batch['num_atoms'][:I])  # I番目の結晶の開始インデックス
        end_index = start_index + batch['num_atoms'][I]  # I番目の結晶の終了インデックス
        first_frac_coords = batch['frac_coords'][start_index:end_index]
        atom_types = batch["atom_types"][start_index:end_index]
        # k1とk2を動かしてcomplex_sumの値を計算
        for j, k1 in enumerate(k1_values):
            for l, k2 in enumerate(k2_values):
                for m, k3 in enumerate(k3_values):
                    k = np.array([k1, k2, k3])
                    Z[I, j, l, m] = I_hkl(k, first_frac_coords, c)
                    # Z[I, j, l, m] = complex_sum_squared(k, first_frac_coords)
                    Z[I, j, l, m] = complex_sum_squared_with_scattering_factors(k, first_frac_coords, c, atom_types)
                    
                    
    return Z

def calculate_for_k_value_m(batch, k1_values, k2_values, k3_values, frac_coords_m, frac_coords_c, delm_coords, num_atoms_max, n_atoms, atom_types):
    batch_results = []
    for k1, k2, k3 in batch:
        K = torch.tensor([k1, k2, k3], dtype=torch.float32, device='cpu')  # CPU上でテンソルを作成
        result = torch.zeros((num_atoms_max, 3), dtype=torch.float32, device='cpu')  # CPU上でテンソルを作成
        for i in range(n_atoms):
            for j in range(n_atoms):
                diff_m = frac_coords_m[i] - frac_coords_m[j]
                sum_c = frac_coords_c[i] + frac_coords_c[j]
                f_i = scattering_factor(atom_types[i], calculate_q_magnitude(K))
                f_j = scattering_factor(atom_types[j], calculate_q_magnitude(K))
                r_m = torch.dot(diff_m, K)
                r_c = torch.dot(sum_c, K**2)
                # temp_result = -4 * np.pi * K * torch.sin(2 * np.pi * r_m) * torch.exp(-2 * np.pi**2 * r_c)
                temp_result = -4 * np.pi * K * f_i * f_j * torch.sin(2 * np.pi * r_m) * torch.exp(-2 * np.pi**2 * r_c)
                result[i] += temp_result * delm_coords[i]
        batch_results.append((k1, k2, k3, result))
    return batch_results

def calculate_for_k_value_c(batch, k1_values, k2_values, k3_values, frac_coords_m, frac_coords_c, delc_coords, num_atoms_max, n_atoms, atom_types):
    batch_results = []
    for k1, k2, k3 in batch:
        K = torch.tensor([k1, k2, k3], dtype=torch.float32, device='cpu')  # CPU上でテンソルを作成
        K_squared = K**2
        result = torch.zeros((num_atoms_max, 3), dtype=torch.float32, device='cpu')  # CPU上でテンソルを作成
        for i in range(n_atoms):
            for j in range(n_atoms):
                diff_m = frac_coords_m[i] - frac_coords_m[j]
                sum_c = frac_coords_c[i] + frac_coords_c[j]
                f_i = scattering_factor(atom_types[i], calculate_q_magnitude(K))
                f_j = scattering_factor(atom_types[j], calculate_q_magnitude(K))
                r_m = np.dot(diff_m, K)
                r_c = np.dot(sum_c, K_squared)
                # temp_result = -4 * np.pi**2 * K_squared * np.cos(2 * np.pi * r_m) * np.exp(-2 * np.pi**2 * r_c)
                temp_result = -4 * np.pi**2 * K_squared * f_i * f_j * np.cos(2 * np.pi * r_m) * np.exp(-2 * np.pi**2 * r_c)
                result[i] += temp_result * delc_coords[i]
        batch_results.append((k1, k2, k3, result))
    return batch_results

def calculate_for_crystal_m(I, k1_values, k2_values, k3_values, num_atoms, m_tensor, c_tensor, delm_delx_t, num_atoms_max, atom_types_all):
    start_index = sum(num_atoms[:I])
    end_index = start_index + num_atoms[I]
    frac_coords_m = m_tensor[start_index:end_index]
    frac_coords_c = c_tensor[start_index:end_index]
    delm_coords = delm_delx_t[start_index:end_index]
    atom_types = atom_types_all[start_index:end_index]
    
    n_atoms = frac_coords_m.shape[0]

    Z = torch.zeros((len(k1_values), len(k2_values), len(k3_values), num_atoms_max, 3), dtype=torch.float32)

    tasks = [(k1, k2, k3) for k1 in k1_values for k2 in k2_values for k3 in k3_values]
    batch_size = max(1, len(tasks) // mp.cpu_count())
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    for batch in batches:
        batch_results = calculate_for_k_value_m(batch, k1_values, k2_values, k3_values, frac_coords_m, frac_coords_c, delm_coords, num_atoms_max, n_atoms, atom_types)
        for k1, k2, k3, result in batch_results:
            a = np.where(k1_values == k1)[0][0]
            b = np.where(k2_values == k2)[0][0]
            c = np.where(k3_values == k3)[0][0]
            Z[a, b, c, :n_atoms, :] = result
    
    return I, Z

def calculate_for_crystal_c(I, k1_values, k2_values, k3_values, num_atoms, m_tensor, c_tensor, delc_delx_t, num_atoms_max, atom_types_all):
    start_index = sum(num_atoms[:I])
    end_index = start_index + num_atoms[I]
    frac_coords_m = m_tensor[start_index:end_index]
    frac_coords_c = c_tensor[start_index:end_index]
    delm_coords = delc_delx_t[start_index:end_index]
    atom_types = atom_types_all[start_index:end_index]
    
    n_atoms = frac_coords_m.shape[0]

    Z = torch.zeros((len(k1_values), len(k2_values), len(k3_values), num_atoms_max, 3), dtype=torch.float32)

    tasks = [(k1, k2, k3) for k1 in k1_values for k2 in k2_values for k3 in k3_values]
    batch_size = max(1, len(tasks) // mp.cpu_count())
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    for batch in batches:
        batch_results = calculate_for_k_value_c(batch, k1_values, k2_values, k3_values, frac_coords_m, frac_coords_c, delm_coords, num_atoms_max, n_atoms, atom_types)
        for k1, k2, k3, result in batch_results:
            a = np.where(k1_values == k1)[0][0]
            b = np.where(k2_values == k2)[0][0]
            c = np.where(k3_values == k3)[0][0]
            Z[a, b, c, :n_atoms, :] = result
    
    return I, Z

# 予測される構造因子のx_t微分を算出(m微分より)
def calculate_delI_delm_delm_delx_t(num_atoms, batch, m_tensor, c_tensor, delm_delx_t, num_workers=4):
    num_atoms = batch['num_atoms'].cpu()
    num_crystals = num_atoms.size(0)
    num_atoms_max = max(num_atoms).cpu()
    m_tensor = m_tensor.cpu()
    c_tensor = c_tensor.cpu()
    atom_types = batch['atom_types'].cpu()
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)
    
    Z = torch.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values), num_atoms_max, 3), dtype=torch.float32)

    # 各結晶の処理を並列化
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(calculate_for_crystal_m, [(I, k1_values, k2_values, k3_values, num_atoms, m_tensor, c_tensor, delm_delx_t, num_atoms_max, atom_types) for I in range(num_crystals)])
        pool.close()  # プールを閉じる
        pool.join()   # すべてのプロセスが終了するのを待つ

    # 結果を統合
    for I, Z_part in results:
        Z[I] = Z_part

    # 最後にZをnumpy.ndarrayに変換
    Z_numpy = Z.numpy()
    
    return Z_numpy

# 予測される構造因子のx_t微分を算出(c微分より)
def calculate_delI_delc_delc_delx_t(num_atoms, batch, m_tensor, c_tensor, delc_delx_t, num_workers=4):
    num_atoms = batch['num_atoms'].cpu()
    num_crystals = num_atoms.size(0)
    num_atoms_max = max(num_atoms).cpu()
    m_tensor = m_tensor.cpu()
    c_tensor = c_tensor.cpu()
    atom_types = batch['atom_types'].cpu()
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)
    
    Z = torch.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values), num_atoms_max, 3), dtype=torch.float32)

    # 各結晶の処理を並列化
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(calculate_for_crystal_c, [(I, k1_values, k2_values, k3_values, num_atoms, m_tensor, c_tensor, delc_delx_t, num_atoms_max, atom_types) for I in range(num_crystals)])
        pool.close()  # プールを閉じる
        pool.join()   # すべてのプロセスが終了するのを待つ

    # 結果を統合
    for I, Z_part in results:
        Z[I] = Z_part

    # 最後にZをnumpy.ndarrayに変換
    Z_numpy = Z.numpy()
    
    return Z_numpy


# 条件スコアを算出
def calculate_dellogp_delx_t(I, y, delI, num_atoms, sigma=0.5):
    # I, y: (n, 5, 5, 5)
    # delI: (n, 5, 5, 5, m, 3)
    # sigma: float
    # num_atoms: list of integers
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.cpu().numpy()

    # Calculate the difference I - y and expand its dimensions to match delI
    difference = (I - y)[:, :, :, :, np.newaxis, np.newaxis]

    # Compute the product (I - y) * delI
    product = difference * delI

    # Sum over the (5, 5, 5) dimensions
    sum_product = np.sum(product, axis=(1, 2, 3))

    # Compute the intermediate result
    intermediate_result = -1 / sigma**2 * sum_product  # shape (n, m, 3)

    # Initialize the final result list
    final_result = []

    # Adjust the result based on num_atoms
    for i, atoms in enumerate(num_atoms):
        final_result.append(intermediate_result[i, :atoms, :])

    # Concatenate the results to form the final output of shape (l, 3)
    final_result = np.concatenate(final_result, axis=0)

    return final_result


def complex_sum_squared(k, A):
    """
    3次元ベクトル k と (n x 3) の行列 A を受け取り、
    |F(h,k,l)|^2 = sum_j sum_k exp{2 * pi * I [(x_j - x_k)h + (y_j - y_k)k + (z_j - z_k)l]}
    を計算する関数。

    Parameters:
    k (np.ndarray): 3次元ベクトル
    A (np.ndarray): (n x 3) の行列

    Returns:
    float: 複素数の和の絶対値の2乗
    """
    i = complex(0, 1)
    pi = np.pi

    # CUDAテンソルをCPUに移動させてNumPy配列に変換
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    # 行数を取得
    n = A.shape[0]

    # 絶対値の2乗の和を計算
    result = 0.0
    for j in range(n):
        for m in range(n):
            diff = A[j] - A[m]
            r = np.dot(diff, k)
            result += np.exp(2 * pi * i * r)

    # 和の絶対値の2乗を計算
    return result


# 真の構造因子と予測される構造因子の差を出す
def calculate_loss(batch, traj, c):
    batch_y = calculate_y(batch, c)
    traj_y = calculate_y(traj, c)
    
    # Ensure the result is a torch tensor
    if not isinstance(batch_y, torch.Tensor):
        batch_y = torch.tensor(batch_y)
    if not isinstance(traj_y, torch.Tensor):
        traj_y = torch.tensor(traj_y)
    
    # RMSEの計算
    mse = torch.nn.functional.mse_loss(batch_y, traj_y)
    rmse = torch.sqrt(mse)
    
    return rmse.item()

# cpuに送る
def to_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor

# ∂(1)/∂mを算出
def calculate_del1_delm(m, c, sigma, x_t):
    m, c, sigma, x_t = map(to_cpu, [m, c, sigma, x_t])
    
    k_values = np.arange(-4, 5).reshape(-1, 1, 1)
    coefficient = - (1 / sigma**2) * (1 / 4) * np.sqrt(2 / (np.pi * c))
    
    Expp = np.exp(-((1/2 + k_values + m - x_t)**2) / (2 * c))
    Expm = np.exp(-((-1/2 + k_values + m - x_t)**2) / (2 * c))
    
    terms_sum = coefficient * (Expm - Expp)
    
    sum_result = np.sum(terms_sum, axis=0)
    
    return sum_result


# ∂(1)/∂cを算出
def calculate_del1_delc(m, c, sigma, x_t):
    m, c, sigma, x_t = map(to_cpu, [m, c, sigma, x_t])
    
    k_values = np.arange(-4, 5).reshape(-1, 1, 1)
    coefficient = -1 / (sigma**2 * 8 * c**(3/2) * np.sqrt(2 * np.pi))
    
    Expp = np.exp(-((1/2 + k_values + m - x_t)**2) / (2 * c))
    Expm = np.exp(-((-1/2 + k_values + m - x_t)**2) / (2 * c))
    
    term1 = (1 + 4 * c) * (Expm - Expp)
    term2 = -2 * (Expm + Expp) * (k_values + m - x_t)
    
    terms_sum = coefficient * (term1 + term2)
    
    sum_result = np.sum(terms_sum, axis=0)
    
    return sum_result


# ∂(2)/∂mを算出
def calculate_del2_delm(m, c, sigma, x_t):
    m, c, sigma, x_t = map(to_cpu, [m, c, sigma, x_t])
    
    k_values = np.arange(-4, 5).reshape(-1, 1, 1)
    coefficient = 1 / (sigma**2 * 2 * c * np.sqrt(2 * np.pi * c))
    
    Expp = np.exp(-((1/2 + k_values + m - x_t)**2) / (2 * c))
    Expm = np.exp(-((-1/2 + k_values + m - x_t)**2) / (2 * c))
    
    #print("Expm:", Expm)
    #print("Expp:", Expp)
    
    term = coefficient * (Expp * (-1/2 - k_values - m + x_t) + Expm * (1/2 - k_values - m + x_t))
    
    #print("Term:", term)
    
    sum_result = np.sum(term, axis=0)
    
    #print("Sum result:", sum_result)
    
    return sum_result

# ∂(2)/∂cを算出
def calculate_del2_delc(m, c, sigma, x_t):
    m, c, sigma, x_t = map(to_cpu, [m, c, sigma, x_t])
    
    k_values = np.arange(-4, 5).reshape(-1, 1, 1)
    coefficient = 1 / (sigma**2 * 4 * c**2 * np.sqrt(2 * np.pi * c))
    
    Expp = np.exp(-((1/2 + k_values + m - x_t)**2) / (2 * c))
    Expm = np.exp(-((-1/2 + k_values + m - x_t)**2) / (2 * c))
    
    #print("Expm:", Expm)
    #print("Expp:", Expp)
    
    term1 = -c * (Expm + Expp)
    term2 = Expp * (1/2 + k_values + m - x_t)**2
    term3 = Expm * (-1/2 + k_values + m - x_t)**2
    
    #print("Term1:", term1)
    #print("Term2:", term2)
    #print("Term3:", term3)
    
    terms_sum = coefficient * (term1 + term2 + term3)
    
    #print("Terms sum:", terms_sum)
    
    sum_result = np.sum(terms_sum, axis=0)
    
    #print("Sum result:", sum_result)
    
    return sum_result

# ∂m/∂x_t, ∂c/∂x_tを算出
def calculate_delx_t(m1, m2, c1, c2, s1, s2):
    m1, m2, c1, c2, s1, s2 = map(to_cpu, [m1, m2, c1, c2, s1, s2])

    # nx3 の行列の形状を取得
    n, _ = m1.shape
    
    # 結果を格納するための配列を初期化
    delm_delx = np.zeros((n, 3))
    delc_delx = np.zeros((n, 3))
    
    # 各要素について delm_delx, delc_delx を計算
    for i in range(n):
        for j in range(3):
            # 定数行列とベクトルを定義
            M = np.array([[m1[i, j], c1[i, j]], [m2[i, j], c2[i, j]]])
            C = np.array([s2[i, j] + m1[i, j], 2 * s2[i, j] * s1[i, j] + m2[i, j]])
            #C = np.array([s2[i, j] + m1[i, j], 0]) #デバッグ用
            
            # 行列方程式を解く
            solution = np.linalg.solve(M, C)
            
            # delm_delx, delc_delx の値を格納
            delm_delx[i, j], delc_delx[i, j] = solution
    
    return delm_delx, delc_delx


# 条件スコアのスケールを変更する
def scale_dellogp_delx_t(batch, dellogp_delx_t, pred_x):
    num_crystals = batch['num_atoms'].size(0)  # バッチサイズ
    final_result = []

    # バッチ内の全ての結晶に対してループ
    for I in range(num_crystals):
        start_index = sum(batch['num_atoms'][:I])  # I番目の結晶の開始インデックス
        end_index = start_index + batch['num_atoms'][I]  # I番目の結晶の終了インデックス
        dellogp_delx_t_per_crystal = dellogp_delx_t[start_index:end_index]
        pred_x_per_crystal = pred_x[start_index:end_index]
        
        # dellogp_delx_t_per_crystalのスケーリング（絶対値の最大値でスケーリング）
        max_dellogp = torch.max(torch.abs(dellogp_delx_t_per_crystal))
        max_pred_x = torch.max(torch.abs(pred_x_per_crystal))
        scaling_factor = max_pred_x / max_dellogp
        scaled_dellogp_delx_t_per_crystal = dellogp_delx_t_per_crystal * scaling_factor

        final_result.append(scaled_dellogp_delx_t_per_crystal)

    # 結果をtorch.Tensorに変換
    final_result = torch.cat(final_result, dim=0)
    
    # dtypeとdeviceをdellogp_delx_tに合わせる
    final_result = final_result.to(dellogp_delx_t.dtype).to(dellogp_delx_t.device)

    return final_result

def scale_dellogp(dellogp_delx_t, pred_x, num_atoms):
    # num_atoms をリストに変換
    if isinstance(num_atoms, torch.Tensor):
        num_atoms = num_atoms.tolist()
    
    # dellogp_delx_t と pred_x を num_atoms に基づいて分割
    split_dellogp_delx_t = torch.split(dellogp_delx_t, num_atoms)
    split_pred_x = torch.split(pred_x, num_atoms)

    scaled_dellogp_delx_t_list = []

    for dellogp, pred in zip(split_dellogp_delx_t, split_pred_x):
        # dellogp_delx_t の各セットの最大絶対値を計算
        max_abs_values_dellogp = dellogp.abs().max(dim=0, keepdim=True)[0]

        # pred_x の各セットの最大絶対値を計算
        max_abs_values_pred = pred.abs().max(dim=0, keepdim=True)[0]

        # pred_x の最大値が dellogp_delx_t の最大値になるようにスケーリング
        scaling_factors = max_abs_values_pred / max_abs_values_dellogp
        scaled_dellogp = dellogp * scaling_factors

        # スケーリングされたテンソルをリストに追加
        scaled_dellogp_delx_t_list.append(scaled_dellogp)

    # スケーリングされたテンソルを元の形状に結合
    return torch.cat(scaled_dellogp_delx_t_list, dim=0)


def log_with_timestamp(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


# スコアから条件スコアを算出する
def calculate_dellogp_delx_t_with_all_flow(x_t, pred_x, pred_x_d2, sigma_x, batch):
    # 新しい方法でmとcを求める
    s1_table, s2_table, m_values, c_values = generate_tables(x_t)
    # log_with_timestamp("1")

    m, c = find_best_fit(s1_table, s2_table, m_values, c_values, pred_x, pred_x_d2, sigma_x, 8)
    # log_with_timestamp("2")

    # yとIを求める
    
    # 真の構造因子
    y = calculate_y(batch, c)
    # log_with_timestamp("3")
    
    # m, cから予測される構造因子を求める
    I = calculate_I(batch, m, c)
    # log_with_timestamp("4")

    # print("真の構造因子(c使用)\n", y.shape, "\n", y[0][0])
    # print("予測されるの構造因子(m, c使用)\n", I.shape, "\n", I[0][0])

    # ∂(1)/∂m, ∂(1)/∂cを求める
    del1_delm = calculate_del1_delm(m, c, sigma_x, x_t)
    # log_with_timestamp("6")
    del1_delc = calculate_del1_delc(m, c, sigma_x, x_t)
    # log_with_timestamp("7")

    # ∂(2)/∂m, ∂(2)/∂cを求める
    del2_delm = calculate_del2_delm(m, c, sigma_x, x_t)
    # log_with_timestamp("8")
    del2_delc = calculate_del2_delc(m, c, sigma_x, x_t)
    # log_with_timestamp("9")

    # ∂m/∂x_t, ∂c/x_tを求める
    delm_delx, delc_delx = calculate_delx_t(del1_delm, del2_delm, del1_delc, del2_delc, pred_x, pred_x_d2)
    # log_with_timestamp("10")

    # ∂I/∂x_tを求める
    delI_delm_delm_delx_t = calculate_delI_delm_delm_delx_t(batch.num_atoms, batch, m, c, delm_delx, 8)
    # log_with_timestamp("11")
    delI_delc_delc_delx_t = calculate_delI_delc_delc_delx_t(batch.num_atoms, batch, m, c, delc_delx, 8)
    # log_with_timestamp("12")

    delI_delx_t = delI_delm_delm_delx_t + delI_delc_delc_delx_t

    dellogp_delx_t = calculate_dellogp_delx_t(I, y, delI_delx_t, batch.num_atoms, sigma=sigma_x)
    # log_with_timestamp("13")
    dellogp_delx_t = torch.tensor(dellogp_delx_t).to('cuda').type(pred_x.dtype)
    # log_with_timestamp("14")
    dellogp_delx_t = scale_dellogp(dellogp_delx_t, pred_x, batch.num_atoms)
    # log_with_timestamp("15")

    return dellogp_delx_t, m, c


def sigma_norm(sigma, T=1.0, sn = 10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T = T)
    return (normal_ ** 2).mean(dim = 0)


class BetaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        scheduler_mode,
        beta_start = 0.0001,
        beta_end = 0.02
    ):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)


        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)

        sigmas[1:] = betas[1:] * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])

        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sigmas', sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)
    

def generate_crystal_structures():
    scale_factor = 0.6
    offset = 0.5 * (1 - scale_factor)

    r1 = np.sqrt(0.5) * scale_factor

    # 正八角形 (octagon)
    octagon = [[(np.cos(theta) * r1 + 1) / 2, (np.sin(theta) * r1 + 1) / 2, 0.5] for theta in np.arange(0, 2 * np.pi, np.pi / 4)]
    octagon = [[x * scale_factor + offset, y * scale_factor + offset, z] for x, y, z in octagon]

    # 直線 (line)
    line = [[0.5, 0.5, (i + 4) * 0.125] for i in range(-4, 4)]
    line = [[x * scale_factor + offset, y * scale_factor + offset, z] for x, y, z in line]

    # 正六面体 (cube)
    square1 = [[(np.cos(theta) * r1 + 1) / 2, (np.sin(theta) * r1 + 1) / 2] for theta in np.arange(np.pi / 4, 2 * np.pi, np.pi / 2)]
    cube = [[x, y, (z + 1) / 2] for z in [-0.5, 0.5] for x, y in square1]
    cube = [[x * scale_factor + offset, y * scale_factor + offset, z * scale_factor + offset] for x, y, z in cube]

    # ジグザグ構造 (zigzag)
    zigzag = [[(x + 1) / 2, (y + 1) / 2, (z + 0.25) * 2] for z in [-0.25, 0.25] for x, y in zip(np.arange(-3 / 8, 0.5, 1 / 4), [-1 / 8, 1 / 8] * 2)]
    zigzag = [[x * scale_factor + offset, y * scale_factor + offset, z * scale_factor + offset] for x, y, z in zigzag]

    structures = [octagon[:8], line[:8], cube[:8], zigzag[:8]] * 2500
    
    return structures

def generate_crystal_structures_2():
    scale_factor = 0.6
    offset = 0.5 * (1 - scale_factor)

    r1 = np.sqrt(0.5) * scale_factor

    # 直線 (line)
    line = [[0.5, 0.5, (i + 4) * 0.125] for i in range(-4, 4)]
    line = [[x * scale_factor + offset, y * scale_factor + offset, z] for x, y, z in line]

    real_sample = [
        [0.3860, 0.9456, 0.0327],
        [0.7741, 0.7231, 0.9265],
        [0.1471, 0.4678, 0.2544],
        [0.1106, 0.3939, 0.5340],
        [0.3488, 0.8717, 0.3123],
        [0.5989, 0.3739, 0.0893],
        [0.7224, 0.6159, 0.6402],
        [0.8970, 0.9656, 0.4775]
    ]

    structures = [line[:8], real_sample] * 5000
    
    return structures

def generate_crystal_structures_3():
    scale_factor = 0.6
    offset = 0.5 * (1 - scale_factor)

    r1 = np.sqrt(0.5) * scale_factor

    # 直線 (line)
    line = [[0.5, 0.5, (i + 4) * 0.125] for i in range(-2, 3)]
    line = [[x * scale_factor + offset, y * scale_factor + offset, z] for x, y, z in line]

    real_sample = [
        [0.00265771, 0.00000000, 0.00000000],
        [0.50015703, 0.50000000, 0.50000000],
        [0.50108143, 0.00000000, 0.50000000],
        [0.50108143, 0.50000000, 0.00000000],
        [0.00050506, 0.50000000, 0.50000000]
    ]

    structures = [line[:5], real_sample] * 5000
    
    return structures

def add_noise_to_structure(structure, std=0.01):
    noisy_structure = []
    for xyz0 in structure:
        xyz1 = np.array(xyz0) + np.random.randn(3) * std
        xyz1 = xyz1 % 1.0  # 座標を0~1の範囲にシフトして、周期境界条件を適用
        noisy_structure.append(xyz1)
    return noisy_structure

class SigmaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        sigma_begin = 0.01,
        sigma_end = 1.0
    ):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)))


        sigmas_norm_ = sigma_norm(sigmas)

        self.register_buffer('sigmas', torch.cat([torch.zeros([1]), sigmas], dim=0))
        self.register_buffer('sigmas_norm', torch.cat([torch.ones([1]), sigmas_norm_], dim=0))

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)


"""
最適化前のm,cを求める式
def find_best_fit(s1_table, s2_table, m_table, c_table, score1, score2, sigma):
    n, d, m_len, c_len = s1_table.shape
    m = np.zeros((n, d))
    c = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            errmin = float('inf')
            kmin = 0
            lmin = 0
            for im in range(m_len):
                for ic in range(c_len):
                    s1 = s1_table[i, j, im, ic]
                    s2 = s2_table[i, j, im, ic]
                    err = (sigma**2 * score1[i, j] - s1)**2 + (sigma**4 * score2[i, j] + sigma**2 - s2)**2
                    if err < errmin:
                        kmin, lmin = im, ic
                        errmin = err
            m[i, j] = m_table[kmin]
            c[i, j] = c_table[lmin]
            #print("m[i,j]", m[i,j])
            #print(i, j, errmin)
    return m, c
"""

"""
m, c最適化 ver.2.0
def calculate_error(i, j, s1_table, s2_table, score1, score2, sigma2, sigma4, m_table, c_table, result_queue):
    # s1_table, s2_table, score1, score2は既にテンソルであるため、そのまま計算
    s1 = s1_table[i, j, :, :]
    s2 = s2_table[i, j, :, :]
    
    # errもテンソルとして計算
    err = (sigma2 * score1[i, j] - s1) ** 2 + (sigma4 * score2[i, j] + sigma2 - s2) ** 2
    
    # errが既にテンソルなので、そのままargminを使用
    min_idx = torch.argmin(err)
    kmin, lmin = divmod(min_idx.item(), s1.shape[1])
    
    # m_tableとc_tableの値を取り出し、結果をキューに格納
    result_queue.put((i, j, m_table[kmin].item(), c_table[lmin].item()))

def find_best_fit(s1_table, s2_table, m_table, c_table, score1, score2, sigma):
    if not isinstance(s1_table, torch.Tensor):
        s1_table = torch.tensor(s1_table).to(sigma.device)
    if not isinstance(s2_table, torch.Tensor):
        s2_table = torch.tensor(s2_table).to(sigma.device)
    if not isinstance(m_table, torch.Tensor):
        m_table = torch.tensor(m_table).to(sigma.device)
    if not isinstance(c_table, torch.Tensor):
        c_table = torch.tensor(c_table).to(sigma.device)

    s1_table = s1_table.cpu()
    s2_table = s2_table.cpu()
    m_table = m_table.cpu()
    c_table = c_table.cpu()
    score1 = score1.cpu()
    score2 = score2.cpu()
    sigma2 = sigma.cpu() ** 2
    sigma4 = sigma.cpu() ** 4

    n, d, m_len, c_len = s1_table.shape
    
    m = torch.zeros((n, d), device=sigma.device)
    c = torch.zeros((n, d), device=sigma.device)

    result_queue = mp.Queue()
    processes = []

    for i in range(n):
        for j in range(d):
            p = mp.Process(target=calculate_error, args=(i, j, s1_table, s2_table, score1, score2, sigma2, sigma4, m_table, c_table, result_queue))
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()

    while not result_queue.empty():
        i, j, m_val, c_val = result_queue.get()
        m[i, j] = m_val
        c[i, j] = c_val

    m = m.cpu()
    c = c.cpu()
    
    return m, c"""


"""
    #この8は後々変える必要がある
    batch_size = batch.num_graphs
    dellogp_delx_t = dellogp_delx_t.view(batch_size, 8, 3)  # 8x3に分割
    pred_x = pred_x.view(batch_size, 8, 3)  # pred_x も 8x3 に分割

    # dellogp_delx_tの各セットの最大絶対値を計算
    max_abs_values_dellogp = dellogp_delx_t.abs().max(dim=1, keepdim=True)[0]

    # pred_xの各セットの最大絶対値を計算
    max_abs_values_pred = pred_x.abs().max(dim=1, keepdim=True)[0]

    # pred_xの最大値の半分がdellogp_delx_tの最大値になるようにスケーリング
    scaling_factors = max_abs_values_pred / max_abs_values_dellogp
    dellogp_delx_t = dellogp_delx_t * scaling_factors

    # テンソルを元の形状 (n, 3) に戻す
    dellogp_delx_t = dellogp_delx_t.view(-1, 3)
    #print("dellogp_delx_t", dellogp_delx_t.shape, "\n", dellogp_delx_t)
    #print("end")
    #print(dellogp_delx_t)
    """

"""def calculate_delI_delm_delm_delx_t(num_atoms, batch, m_tensor, c_tensor, delm_delx_t):
    num_crystals = batch['num_atoms'].size(0)  # バッチサイズ
    num_atoms_max = max(num_atoms)  # 最大のnum_atomsを持つ結晶の数

    # kの範囲設定
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)

    # 結果を格納する配列
    Z = np.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values), num_atoms_max, 3))

    # バッチ内の全ての結晶に対してループ
    for I in range(num_crystals):
        start_index = sum(batch['num_atoms'][:I])  # I番目の結晶の開始インデックス
        end_index = start_index + batch['num_atoms'][I]  # I番目の結晶の終了インデックス
        frac_coords_m = m_tensor[start_index:end_index]
        frac_coords_c = c_tensor[start_index:end_index]
        delm_coords = delm_delx_t[start_index:end_index]
        
        n_atoms = frac_coords_m.shape[0]  # 現在の結晶の原子数

        # k1とk2を動かして複素数の和を計算
        for a, k1 in enumerate(k1_values):
            for b, k2 in enumerate(k2_values):
                for c, k3 in enumerate(k3_values):
                    K = np.array([k1, k2, k3])
                    # 計算を行う
                    result = np.zeros((num_atoms_max, 3))
                    for i in range(n_atoms):
                        for j in range(n_atoms):
                            diff_m = frac_coords_m[i] - frac_coords_m[j]
                            sum_c = frac_coords_c[i] + frac_coords_c[j]
                            r_m = np.dot(diff_m, K)
                            r_c = np.dot(sum_c, K**2)
                            temp_result = -4 * np.pi * K * np.sin(2 * np.pi * r_m) * np.exp(-2 * np.pi**2 * r_c)
                            result[i] += temp_result * delm_coords[i] 
                            # result[i] += temp_result #一旦∂I/∂mを求める
                    Z[I, a, b, c, :n_atoms, :] = result

    return Z"""

"""
def calculate_delI_delm_delm_delx_t(num_atoms, batch, m_tensor, c_tensor, delm_delx_t, num_workers=4):
    num_crystals = batch['num_atoms'].size(0)
    num_atoms_max = max(num_atoms).cpu()
    
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)
    
    Z = torch.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values), num_atoms_max, 3), dtype=torch.float32)

    for I in range(num_crystals):
        start_index = sum(batch['num_atoms'][:I])
        end_index = start_index + batch['num_atoms'][I]
        frac_coords_m = m_tensor[start_index:end_index].cpu()
        frac_coords_c = c_tensor[start_index:end_index].cpu()
        delm_coords = delm_delx_t[start_index:end_index]
        
        n_atoms = frac_coords_m.shape[0]

        # (k1, k2, k3) の組み合わせをすべて列挙
        tasks = [(k1, k2, k3) for k1 in k1_values for k2 in k2_values for k3 in k3_values]

        # タスクをバッチに分割
        batch_size = max(1, len(tasks) // num_workers)
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        # print([(batch, k1_values, k2_values, k3_values, frac_coords_m, frac_coords_c, delm_coords, num_atoms_max, n_atoms) for batch in batches])
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(calculate_for_k_value, [(batch, k1_values, k2_values, k3_values, frac_coords_m, frac_coords_c, delm_coords, num_atoms_max, n_atoms) for batch in batches])
        
        # 結果を Z に反映
        for batch_results in results:
            for k1, k2, k3, result in batch_results:
                a = np.where(k1_values == k1)[0][0]
                b = np.where(k2_values == k2)[0][0]
                c = np.where(k3_values == k3)[0][0]
                Z[I, a, b, c, :n_atoms, :] = result
    
    return Z"""

"""def calculate_delI_delc_delc_delx_t(num_atoms, batch, m_tensor, c_tensor, delc_delx_t):
    num_crystals = batch['num_atoms'].size(0)  # バッチサイズ
    num_atoms_max = max(num_atoms)  # 最大のnum_atomsを持つ結晶の数

    # kの範囲設定
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)

    # 結果を格納する配列
    Z = np.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values), num_atoms_max, 3))

    # バッチ内の全ての結晶に対してループ
    for I in range(num_crystals):
        start_index = sum(batch['num_atoms'][:I])  # I番目の結晶の開始インデックス
        end_index = start_index + batch['num_atoms'][I]  # I番目の結晶の終了インデックス
        frac_coords_m = m_tensor[start_index:end_index]
        frac_coords_c = c_tensor[start_index:end_index]
        delc_coords = delc_delx_t[start_index:end_index]
        
        n_atoms = frac_coords_m.shape[0]  # 現在の結晶の原子数

        # k1とk2を動かして複素数の和を計算
        for a, k1 in enumerate(k1_values):
            for b, k2 in enumerate(k2_values):
                for c, k3 in enumerate(k3_values):
                    K = np.array([k1, k2, k3])
                    K_squared = K**2  # 各要素の二乗からなる3次元ベクトル
                    # 計算を行う
                    result = np.zeros((num_atoms_max, 3))
                    for i in range(n_atoms):
                        for j in range(n_atoms):
                            diff_m = frac_coords_m[i] - frac_coords_m[j]
                            sum_c = frac_coords_c[i] + frac_coords_c[j]
                            r_m = np.dot(diff_m, K)
                            r_c = np.dot(sum_c, K_squared)
                            temp_result = -4 * np.pi**2 * K_squared * np.cos(2 * np.pi * r_m) * np.exp(-2 * np.pi**2 * r_c)
                            result[i] += temp_result * delc_coords[i]
                            # result[i] += temp_result
                    Z[I, a, b, c, :n_atoms, :] = result
    return Z"""