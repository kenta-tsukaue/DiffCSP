import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.integrate import quad
import math
from torch.autograd import Variable


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
        p_ += torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return p_

def log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    sum_exp = torch.zeros_like(x)
    for i in range(-N, N+1):
        sum_exp += torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    log_p = torch.log(sum_exp)
    return log_p

def d_log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    p_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        p_ += (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return p_ / p_wrapped_normal_sampling(x, sigma, N, T)

def d_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    dp_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        dp_ += (-(x + T * i) / (sigma ** 2)) * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return dp_

def d2_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    d2p_ = torch.zeros_like(x)
    for i in range(-N, N+1):
        d2p_ += (((x + T * i)**2 / (sigma**4)) - (1 / (sigma**2))) * torch.exp(-(x + T * i) ** 2 / (2 * sigma ** 2))
    return d2p_

def d2_log_p_wrapped_normal_sampling(x, sigma, N=10, T=1.0):
    p = p_wrapped_normal_sampling(x, sigma, N, T)
    dp = d_p_wrapped_normal_sampling(x, sigma, N, T)
    d2p = d2_p_wrapped_normal_sampling(x, sigma, N, T)
    d2_log_p = (d2p * p - dp**2) / p**2
    return d2_log_p

# フーリエ係数の計算関数(an)
def compute_fourier_an(n, sigma, N=10, T=1.0, num_points=1000):
    # 積分区間 [0, T] を num_points 個の点で離散化
    x = torch.linspace(0, T, num_points, requires_grad=True)
    dx = T / num_points
    
    # 導関数を評価
    f_x = log_p_wrapped_normal_sampling(x, sigma, N, T)
    
    # コサイン成分の計算
    cos_term = torch.cos(2 * torch.pi * n * x)
    integral = torch.sum(f_x * cos_term) * dx
    
    # 係数 a_n の計算
    a_n = 2 * integral  # 0 以外の n に対しては 2 を乗算
    return a_n

def compute_fourier_bn(n, sigma, N=10, T=1.0, num_points=1000):
    # 積分区間 [0, T] を num_points 個の点で離散化
    x = torch.linspace(0, T, num_points, requires_grad=True)
    dx = T / num_points
    
    # 導関数を評価
    f_x = d_log_p_wrapped_normal_sampling(x, sigma, N, T)**2 + d2_log_p_wrapped_normal_sampling(x, sigma, N, T)
    
    # コサイン成分の計算
    # コサイン成分の計算（n=0 の場合は単に 1）
    if n == 0:
        cos_term = torch.ones(num_points)  # n=0 の場合、cos(0) = 1
    else:
        cos_term = torch.cos(2 * torch.pi * n * x)
    integral = torch.sum(f_x * cos_term) * dx

    # 係数 b_n の計算（n=0 の場合は係数 1、それ以外は 2）
    if n == 0:
        b_n = integral  # n=0 の場合、係数は 1
    else:
        b_n = 2 * integral  # 0 以外の n に対しては 2 を乗算
    
    return b_n 

def optimize_mc(x_t, sigma, target1, target2, lr=0.01, iterations=1000):
    
    """
    target1: 式1のターゲット値
    target2: 式2のターゲット値
    compute_fourier_an: a_n を計算する関数
    compute_fourier_bn: b_n を計算する関数
    lr: 学習率
    iterations: 最適化の反復回数
    """
    # 初期値
     # mとcの初期化
    m = Variable(torch.randn(x_t.shape), requires_grad=True)
    c = Variable(torch.randn(x_t.shape), requires_grad=True)

    print(m.size())
    print(c.size())

    # オプティマイザー
    optimizer = torch.optim.Adam([m, c], lr=lr)
    # a_n, b_n の計算
    a_n_values = torch.tensor([compute_fourier_an(n, sigma) for n in range(1, 6)]).view(-1, 1, 1)
    b_n_values = torch.tensor([compute_fourier_bn(n, sigma) for n in range(0, 6)]).view(-1, 1, 1)


    # 最適化ループ
    for _ in range(iterations):
        optimizer.zero_grad()

        # 式の計算
        sum_expr_1 = torch.sum(a_n_values * torch.sin(2 * torch.pi * torch.arange(1, 6).view(-1, 1, 1) * m) * torch.exp(-(2 * torch.pi * torch.arange(1, 6).view(-1, 1, 1))**2 * c / 2), dim=0)
        sum_expr_2 = b_n_values[0]/2 + torch.sum(b_n_values[1:] * torch.cos(2 * torch.pi * torch.arange(1, 6).view(-1, 1, 1) * m) * torch.exp(-(2 * torch.pi * torch.arange(1, 6).view(-1, 1, 1))**2 * c / 2), dim=0)
        
        print(sum_expr_1.size())
        print(sum_expr_2.size())
        print(target1.size())
        print(target2.size())

        # 損失関数
        loss = (sum_expr_1 - target1)**2 + (sum_expr_2 - target2)**2
        loss = loss.sum()  # 全体の損失を合計
        
        # バックプロパゲーション
        loss.backward()
        optimizer.step()

    return m.detach(), c.detach()

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


