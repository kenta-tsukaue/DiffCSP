import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.integrate import quad
import math
from torch.autograd import Variable
from scipy.optimize import minimize



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
    p_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        p_ = p_ + torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return p_

def log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    sum_exp = torch.zeros_like(x)
    for i in range(-N, N+1):
        sum_exp = sum_exp + torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    log_p = torch.log(sum_exp)
    return log_p

def d_log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    p_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        p_ = p_ + (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return p_ / p_wrapped_normal_sampling(x, sigma, N, T)

def d_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    dp_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        dp_ = dp_ + (-(x + T * i) / (sigma ** 2)) * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return dp_

def d2_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    d2p_ = torch.zeros_like(x)
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
    x = torch.linspace(0, T, num_points)
    dx = T / num_points
    f_x = log_p_wrapped_normal_sampling(x, sigma, N, T)
    cos_term = torch.cos(2 * torch.pi * n * x)
    integral = torch.sum(f_x * cos_term) * dx
    a_n = 2 * integral
    return a_n

def compute_fourier_bn(n, sigma, N=10, T=1.0, num_points=1000):
    x = torch.linspace(0, T, num_points)
    dx = T / num_points
    f_x = d_log_p_wrapped_normal_sampling(x, sigma, N, T) ** 2 + d2_log_p_wrapped_normal_sampling(x, sigma, N, T)
    if n == 0:
        cos_term = torch.ones_like(x)
    else:
        cos_term = torch.cos(2 * torch.pi * n * x)
    integral = torch.sum(f_x * cos_term) * dx
    b_n = integral if n == 0 else 2 * integral
    return b_n

def loss_function(params, x_t, a_n_values, b_n_values, target1, target2):
    m = torch.tensor(params[:x_t.numel()].reshape(x_t.shape), dtype=torch.float32)
    c = torch.tensor(params[x_t.numel():].reshape(x_t.shape), dtype=torch.float32)

    exp_terms = torch.exp(-(2 * torch.pi * torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1)) ** 2 * c / 2)
    exp_terms = torch.clamp(exp_terms, min=1e-10, max=1e10)

    sum_expr_1 = torch.sum(a_n_values * torch.sin(2 * torch.pi * torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * m) * exp_terms, dim=0)

    exp_terms_b = torch.exp(-(2 * torch.pi * torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1)) ** 2 * c / 2)
    exp_terms_b = torch.clamp(exp_terms_b, min=1e-15, max=1e15)

    sum_expr_2 = b_n_values[0] / 2 + torch.sum(b_n_values[1:] * torch.cos(2 * torch.pi * torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * m) * exp_terms_b, dim=0)

    loss = ((sum_expr_1 - target1) ** 2 + (sum_expr_2 - target2) ** 2).sum()

    return loss.item()

def optimize_mc_scipy(x_t, sigma, target1, target2, iterations=1000):
    # 初期値の設定
    x_t = x_t.clone().detach().requires_grad_(True)  # Ensure x_t requires gradient
    initial_m = np.random.randn(*x_t.shape) * 0.1
    initial_c = np.random.randn(*x_t.shape) * 0.1
    initial_params = np.concatenate([initial_m.flatten(), initial_c.flatten()])
    a_n_values = torch.stack([compute_fourier_an(n, sigma) for n in range(1, 6)], dim=0).view(-1, 1, 1)
    b_n_values = torch.stack([compute_fourier_bn(n, sigma) for n in range(0, 6)], dim=0).view(-1, 1, 1)

    # 制約を設定
    lower_bound_c = 1e-6  # cの下限を設定
    bounds = [(None, None)] * x_t.numel() + [(lower_bound_c, None)] * x_t.numel()

    # 最適化の実行
    result = minimize(loss_function, initial_params, args=(x_t, a_n_values, b_n_values, target1, target2), 
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': iterations})

    # 最適化されたパラメータの取得
    optimized_params = result.x
    m_optimized = optimized_params[:x_t.numel()].reshape(x_t.shape)
    c_optimized = optimized_params[x_t.numel():].reshape(x_t.shape)

    # NumPy 配列を PyTorch テンソルに変換
    m_optimized = torch.from_numpy(m_optimized).float().requires_grad_(True)
    c_optimized = torch.from_numpy(c_optimized).float().requires_grad_(True)

    # Compute gradients for each element independently
    m_optimized_grad = torch.zeros_like(m_optimized)
    c_optimized_grad = torch.zeros_like(c_optimized)

    for i in range(m_optimized.shape[0]):
        for j in range(m_optimized.shape[1]):
            # Compute gradient for each component independently
            m_optimized_grad[i, j] = torch.autograd.grad(outputs=m_optimized[i, j], inputs=x_t, create_graph=True, retain_graph=True)[0]
            c_optimized_grad[i, j] = torch.autograd.grad(outputs=c_optimized[i, j], inputs=x_t, create_graph=True, retain_graph=True)[0]

    return m_optimized, c_optimized, m_optimized_grad, c_optimized_grad

def calculate_structure_factors(m, c, num_atoms, m_grad, c_grad):
    # Qベクトルの生成
    Q_values = np.array([0, 1, 2, 3, 4])
    Q_combinations = np.array(np.meshgrid(Q_values, Q_values, Q_values)).T.reshape(-1, 3)
    
    # 結晶の数
    num_crystals = len(num_atoms)

    z = complex(0, 1)
    
    # 出力の初期化
    structure_factors = np.zeros((num_crystals, 5, 5, 5))
    derivative_of_SQ = np.zeros_like(structure_factors)  # Initialize the derivative array
    
    # 開始インデックス
    start_index = 0
    
    for crystal_index, num_atom in enumerate(num_atoms):
        # 結晶の各原子に対するmとcのサブセット
        m_subset = m[start_index:start_index + num_atom]
        c_subset = c[start_index:start_index + num_atom]
        m_grad_subset = m_grad[start_index:start_index + num_atom]
        c_grad_subset = c_grad[start_index:start_index + num_atom]
        
        # 各Qに対する構造因子の計算
        for i, Q in enumerate(Q_combinations):
            sum_SQ = 0
            sum_dSQ = 0  # Sum for the derivative
            for j in range(num_atom):
                for k in range(num_atom):
                    term1 = z * np.dot(Q, m_subset[k] - m_subset[j])
                    term2 = -0.5 * np.dot(c_subset[k] + c_subset[j], Q ** 2) 
                    exp_term = np.exp(term1 + term2)
                    sum_SQ += exp_term
                    
                    # Derivative terms
                    d_term1 = z * np.dot(Q, m_grad_subset[k] - m_grad_subset[j])
                    d_term2 = -0.5 * np.dot(c_grad_subset[k] + c_grad_subset[j], Q ** 2)
                    sum_dSQ += exp_term * (d_term1 + d_term2)
            
            # 結果の格納（絶対値を保存）
            structure_factors[crystal_index, i // 25, (i % 25) // 5, (i % 25) % 5] = np.abs(sum_SQ)
            derivative_of_SQ[crystal_index, i // 25, (i % 25) // 5, (i % 25) % 5] = np.abs(sum_dSQ)
        
        # 次の結晶のためにインデックスを更新
        start_index += num_atom

    return structure_factors, derivative_of_SQ



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
