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
from scipy.special import erf
from scipy.constants import pi

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
    return d2_log_p


def p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    p_ = torch.zeros_like(x,device=x.device)
    for i in range(-N, N+1):
        p_ = p_ + torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return p_

def log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    sum_exp = torch.zeros_like(x,device=x.device)
    for i in range(-N, N+1):
        sum_exp = sum_exp + torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    log_p = torch.log(sum_exp)
    return log_p

def d_log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    p_ = torch.zeros_like(x,device=x.device)
    for i in range(-N, N+1):
        p_ = p_ + (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return p_ / p_wrapped_normal_sampling(x, sigma, N, T)

def d_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    dp_ = torch.zeros_like(x,device=x.device)
    for i in range(-N, N+1):
        dp_ = dp_ + (-(x + T * i) / (sigma ** 2)) * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return dp_

def d2_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    d2p_ = torch.zeros_like(x,device=x.device)
    for i in range(-N, N+1):
        d2p_ = d2p_ + (((x + T * i)**2 / (sigma**4)) - (1 / (sigma**2))) * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return d2p_

def d2_log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    p = p_wrapped_normal_sampling(x, sigma, N, T)
    dp = d_p_wrapped_normal_sampling(x, sigma, N, T)
    d2p = d2_p_wrapped_normal_sampling(x, sigma, N, T)
    d2_log_p = (d2p * p - dp**2) / p**2
    return d2_log_p

# フーリエ係数の計算関数(an)
def compute_fourier_an(n, sigma, N=10, T=1.0, num_points=1000):
    device = "cpu"  # sigmaのデバイスを取得
    x = torch.linspace(0, T, num_points, device=device)
    dx = T / num_points
    f_x = log_p_wrapped_normal_sampling(x, sigma, N, T)
    cos_term = torch.cos(2 * torch.pi * n * x)
    integral = torch.sum(f_x * cos_term) * dx
    a_n = 2 * integral
    return a_n

def compute_fourier_bn(n, sigma, N=10, T=1.0, num_points=1000):
    device = "cpu"
    x = torch.linspace(0, T, num_points, device=device)
    dx = T / num_points
    f_x = d_log_p_wrapped_normal_sampling(x, sigma, N, T) ** 2 + d2_log_p_wrapped_normal_sampling(x, sigma, N, T)
    if n == 0:
        cos_term = torch.ones_like(x)
    else:
        cos_term = torch.cos(2 * torch.pi * n * x)
    integral = torch.sum(f_x * cos_term) * dx
    b_n = integral if n == 0 else 2 * integral
    return b_n

def loss_function_scipy(params, x_t, a_n_values, b_n_values, target1, target2):
    m = params[:x_t.size].reshape(x_t.shape)
    c = params[x_t.size:].reshape(x_t.shape)

    exp_terms = np.exp(-(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1)) ** 2 * c / 2)
    exp_terms = np.clip(exp_terms, 1e-32, 1e32)

    sum_expr_1 = np.sum(a_n_values * np.sin(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1) * m) * exp_terms, axis=0)

    exp_terms_b = np.exp(-(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1)) ** 2 * c / 2)
    exp_terms_b = np.clip(exp_terms_b, 1e-32, 1e32)

    sum_expr_2 = b_n_values[0] / 2 + np.sum(b_n_values[1:] * np.cos(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1) * m) * exp_terms_b, axis=0)

    # target1 と target2 を numpy array に変換
    target1_np = target1.cpu().detach().numpy()
    target2_np = target2.cpu().detach().numpy()

    loss = ((sum_expr_1 - target1_np) ** 2 + (sum_expr_2 - target2_np) ** 2).sum()
    #print("total_loss", loss)
    #print("sum_expr_1_loss", ((sum_expr_1 - target1_np) ** 2).sum())
    #print("sum_expr_2_loss", ((sum_expr_2 - target2_np) ** 2).sum())

    return loss

# 最適化関数の定義
def optimize_mc(x_t, sigma, target1, target2, iterations=20):
    # ロスを記録するリスト
    loss_history = []

    # コールバック関数の定義
    def callback(params):
        iteration = len(loss_history) + 1
        m = params[:x_t.numel()].reshape(x_t.shape)
        c = params[x_t.numel():].reshape(x_t.shape)
        loss = loss_function_scipy(params, x_t_cpu, a_n_values, b_n_values, target1, target2)
        loss_history.append(loss)
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Current loss: {loss}")
    
    x_t_cpu = x_t.cpu().detach().numpy()
    sigma_cpu = sigma.cpu().detach().numpy()
    initial_m = np.ones_like(x_t_cpu) * 0.5
    initial_c = np.ones_like(x_t_cpu) * 0.001
    initial_params = np.concatenate([initial_m.flatten(), initial_c.flatten()])
    a_n_values = np.stack([compute_fourier_an(n, sigma_cpu) for n in range(1, 6)], axis=0).reshape(-1, 1, 1)
    b_n_values = np.stack([compute_fourier_bn(n, sigma_cpu) for n in range(0, 6)], axis=0).reshape(-1, 1, 1)

    lower_bound_c = 1e-32  # cの下限を設定
    bounds = [(None, None)] * x_t_cpu.size + [(lower_bound_c, None)] * x_t_cpu.size

    """result = minimize(loss_function_scipy, initial_params, args=(x_t_cpu, a_n_values, b_n_values, target1, target2), 
                  method='TNC', bounds=bounds, options={'maxiter': iterations, 'ftol': 1e-9}, callback=callback)"""
    result = minimize(loss_function_scipy, initial_params, args=(x_t_cpu, a_n_values, b_n_values, target1, target2), 
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': iterations}, callback=callback)
    """result = minimize(loss_function_sol, initial_params, args=(x_t_cpu, sigma_cpu, target1, target2), 
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': iterations}, callback=callback)"""
    #L-BFGS-B
    optimized_params = result.x
    m_optimized = optimized_params[:x_t_cpu.size].reshape(x_t_cpu.shape)
    c_optimized = optimized_params[x_t_cpu.size:].reshape(x_t_cpu.shape)

    return m_optimized, c_optimized

def check_sol_2(m, c, x_t, sigma, target1, target2, sol_1, sol_2):
    sigma_cpu = sigma.cpu().detach().numpy()
    a_n_values = np.stack([compute_fourier_an(n, sigma_cpu) for n in range(1, 6)], axis=0).reshape(-1, 1, 1)
    b_n_values = np.stack([compute_fourier_bn(n, sigma_cpu) for n in range(0, 6)], axis=0).reshape(-1, 1, 1)
    exp_terms = np.exp(-(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1)) ** 2 * c / 2)
    exp_terms = np.clip(exp_terms, 1e-32, 1e32)

    sum_expr_1 = np.sum(a_n_values * np.sin(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1) * m) * exp_terms, axis=0)

    exp_terms_b = np.exp(-(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1)) ** 2 * c / 2)
    exp_terms_b = np.clip(exp_terms_b, 1e-32, 1e32)

    sum_expr_2 = b_n_values[0] / 2 + np.sum(b_n_values[1:] * np.cos(2 * np.pi * np.arange(1, 6).reshape(-1, 1, 1) * m) * exp_terms_b, axis=0)

    print("==================[target1=================\n", target1)
    print("==================[sum_expr_1]=================\n", sum_expr_1)
    print("==================[sol_1]=================\n", sol_1)
    print("==================[target2]=================\n", target2)
    print("==================[sum_expr_2]=================\n", sum_expr_2)
    print("==================[sol_2]=================\n", sol_2)
    print("==================[m]=================\n", m)
    print("==================[c]=================\n", c)
    


# target1 と target2 を計算する関数
def calculate_targets(decoder, decoder_d2, time_emb, atom_types, x_t, l_t, num_atoms, batch):
    pred_l, pred_x = decoder(time_emb, atom_types, x_t, l_t, num_atoms, batch)
    _, pred_x_d2 = decoder_d2(time_emb, atom_types, x_t, l_t, num_atoms, batch)
    target1 = pred_x
    target2 = pred_x_d2 + pred_x ** 2
    return target1, target2

def calculate_derivatives(decoder, decoder_d2, time_emb, atom_types, x_t, l_t, num_atoms, batch, sigma, iterations=1000):
    target1, target2 = calculate_targets(decoder, decoder_d2, time_emb, atom_types, x_t, l_t, num_atoms, batch)
    m_optimized, c_optimized = optimize_mc(x_t, sigma, target1, target2, iterations)

    def optimized_m(x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=x_t.device)
        target1, target2 = calculate_targets(decoder, decoder_d2, time_emb, atom_types, x_tensor, l_t, num_atoms, batch)
        return optimize_mc(x_tensor, sigma, target1, target2, iterations)[0]

    def optimized_c(x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=x_t.device)
        target1, target2 = calculate_targets(decoder, decoder_d2, time_emb, atom_types, x_tensor, l_t, num_atoms, batch)
        return optimize_mc(x_tensor, sigma, target1, target2, iterations)[1]

    x_t_cpu = x_t.cpu().detach().numpy()  # x_t を CPU に移動させて NumPy 配列に変換
    dm_dx_t = derivative(optimized_m, x_t_cpu, dx=1e-2)
    dc_dx_t = derivative(optimized_c, x_t_cpu, dx=1e-2)

    return m_optimized, c_optimized, dm_dx_t, dc_dx_t


"""
=======================================
最新手法 m, c の計算 & table作成
=======================================
"""
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
    c_values = np.arange(1e-2, 2 + 1/10, 1/10)
    
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
            print(i, j, errmin)
    return m, c

def S_Q(x_t, m, c, num_atoms):
    Q = np.array(np.meshgrid(np.arange(5), np.arange(5), np.arange(5))).T.reshape(-1, 3)  # 3次元の Q ベクトル
    S_list = []
    start = 0
    for atoms in num_atoms:
        end = start + atoms
        S = np.zeros(Q.shape[:-1], dtype=complex)  # Q の形状に合わせたゼロ初期化
        for i in range(start, end):
            for j in range(start, end):
                f_ij = 1j * np.sum(Q * (m[j] - m[i]), axis=-1) - 0.5 * np.sum((Q**2) * (c[j] + c[i]), axis=-1)
                S += np.exp(f_ij)  # 各次元での合計を取る
        S_list.append(np.abs(S))
        start = end
    return np.array(S_list).reshape(len(num_atoms), 5, 5, 5)


def dS_Q_dx_t(x_t, m, c, dm_dx_t, dc_dx_t, num_atoms):
    Q = np.array(np.meshgrid(np.arange(5), np.arange(5), np.arange(5))).T.reshape(-1, 3)  # 3次元の Q ベクトル
    num_crystals = len(num_atoms)
    max_atoms = max(num_atoms)
    dS_dx_t_list = np.zeros((num_crystals, 5, 5, 5, max_atoms, 3), dtype=complex)

    start = 0
    for crystal_index, atoms in enumerate(num_atoms):
        end = start + atoms
        for i in range(start, end):
            for j in range(start, end):
                f_ij = 1j * np.sum(Q * (m[j] - m[i]), axis=-1) - 0.5 * np.sum((Q**2) * (c[j] + c[i]), axis=-1)
                exp_f_ij = np.exp(f_ij).reshape(5, 5, 5)  # 各次元での合計を取る
                for k in range(3):  # 各次元成分について計算
                    df_ij_dx_t = 1j * Q[:, k] * (dm_dx_t[j, k] - dm_dx_t[i, k]) - 0.5 * Q[:, k]**2 * (dc_dx_t[j, k] + dc_dx_t[i, k])
                    df_ij_dx_t = df_ij_dx_t.reshape(5, 5, 5)  # 形状を合わせる
                    dS_dx_t_list[crystal_index, :, :, :, i-start, k] += exp_f_ij * df_ij_dx_t
        start = end
    
    return np.abs(dS_dx_t_list)

def calculate_dy(dS_dx_t_result, y, s, num_atoms):
    sigma = 100000
    num_crystals = len(num_atoms)
    n = sum(num_atoms)
    expression_result = np.zeros((n, 3), dtype=complex)
    
    start = 0
    for crystal_index, atoms in enumerate(num_atoms):
        end = start + atoms
        crystal_y = y[crystal_index]
        crystal_s = s[crystal_index]

        for i in range(atoms):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        expression_result[start + i, :] += -1 * (crystal_y[j, k, l] - crystal_s[j, k, l]) * dS_dx_t_result[crystal_index, j, k, l, i, :] / sigma
        start = end
    expression_result = expression_result.real
    
    return expression_result


def complex_sum(k, A):
    """
    3次元ベクトル k と (n x 3) の行列 A を受け取り、
    行列 A の各行とベクトル k の内積 r を計算し、
    exp(2 * pi * i * r) を足し合わせて、その和を返す関数。

    Parameters:
    k (np.ndarray): 3次元ベクトル
    A (np.ndarray): (n x 3) の行列

    Returns:
    complex: 複素数の和
    """
    i = complex(0, 1)
    pi = np.pi

    # CUDAテンソルをCPUに移動させてNumPy配列に変換
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    # 各行とベクトル k の内積を計算
    r = np.dot(A, k)

    # exp(2 * pi * i * r) の和を計算
    result = np.sum(np.exp(2 * pi * i * r))

    return result

def calculate_y(num_atoms, batch):
    num_crystals = batch['num_atoms'].size(0)  # バッチサイズ
    # kの範囲設定
    k1_values = np.arange(0, 5, 1)
    k2_values = np.arange(0, 5, 1)
    k3_values = np.arange(0, 5, 1)
    # 結果を格納する配列
    Z = np.zeros((num_crystals, len(k1_values), len(k2_values), len(k3_values)))

    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
        start_index = sum(batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        end_index = start_index + batch['num_atoms'][i]  # i番目の結晶の終了インデックス
        first_frac_coords = batch['frac_coords'][start_index:end_index]
        # k1とk2を動かしてcomplex_sumの値を計算
        for j, k1 in enumerate(k1_values):
            for l, k2 in enumerate(k2_values):
                for m, k3 in enumerate(k3_values):
                    k = np.array([k1, k2, k3])
                    Z[i, j, l, m] = np.abs(complex_sum(k, first_frac_coords))
    return Z
    


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
