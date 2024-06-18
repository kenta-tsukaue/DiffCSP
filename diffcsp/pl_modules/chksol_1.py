#  ある自由度で、方程式の解(m,c)を数値積分でチェックする
import numpy as np
import torch
from scipy.integrate import quad
from concurrent.futures import ThreadPoolExecutor


# G[x]とP[x]の定義
def G(x, sigma):
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    sigma = sigma.cpu().numpy() if isinstance(sigma, torch.Tensor) else sigma
    summation = sum(np.exp(-(x + k)**2 / (2 * sigma**2)) for k in range(-3,4))
    return summation / np.sqrt(2 * np.pi * sigma**2) 

def P(x, c, m):
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    c = c.cpu().numpy() if isinstance(c, torch.Tensor) else c
    m = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
    summation = sum(np.exp(-(x - m + l)**2 / (2 * c)) for l in range(-3,4))
    return summation / np.sqrt(2 * np.pi * c) 

# d/dx[log(G[x])]の定義
def d_log_G(x, sigma):
    G_val = G(x, sigma)
    if G_val == 0:
        return 0  # G_valがゼロの場合の処理
    dG_dx = sum(-(x + k) * np.exp(-(x + k)**2 / (2 * sigma**2)) / (sigma**2) for k in range(-3,4))
    dG_dx /= np.sqrt(2 * np.pi * sigma**2) 
    return dG_dx / G_val

# d2/dx2[log(G[x])]の定義
def d2_log_G(x, sigma):
    G_val = G(x, sigma)
    if G_val == 0:
        return 0  # G_valがゼロの場合の処理
    dG_dx = sum(-(x + k) * np.exp(-(x + k)**2 / (2 * sigma**2)) / (sigma**2) for k in range(-3,4))
    dG_dx /= np.sqrt(2 * np.pi * sigma**2)
    d2G_dx2 = sum(((x + k)**2 / sigma**4 - 1 / sigma**2) * np.exp(-(x + k)**2 / (2 * sigma**2)) for k in range(-3,4))
    d2G_dx2 /= np.sqrt(2 * np.pi * sigma**2)
    return (d2G_dx2 * G_val - dG_dx**2) / G_val**2

# 被積分関数の定義
def integrand1(x, sigma, c, m):
    sigma = sigma.item() if isinstance(sigma, (np.ndarray, torch.Tensor)) and sigma.size == 1 else sigma
    c = c.item() if isinstance(c, (np.ndarray, torch.Tensor)) and c.size == 1 else c
    m = m.item() if isinstance(m, (np.ndarray, torch.Tensor)) and m.size == 1 else m
    return d_log_G(x, sigma) * P(x, c, m)

def integrand2(x, sigma, c, m):
    sigma = sigma.item() if isinstance(sigma, (np.ndarray, torch.Tensor)) and sigma.size == 1 else sigma
    c = c.item() if isinstance(c, (np.ndarray, torch.Tensor)) and c.size == 1 else c
    m = m.item() if isinstance(m, (np.ndarray, torch.Tensor)) and m.size == 1 else m
    return (d2_log_G(x, sigma) + (d_log_G(x, sigma))**2) * P(x, c, m)

# 数値積分
def integrate(sigma, c, m):
    def compute_integrals(i, j):
        result1, error1 = quad(integrand1, 0, 1, args=(sigma, c[i, j], m[i, j]))
        result2, error2 = quad(integrand2, 0, 1, args=(sigma, c[i, j], m[i, j]))
        return (i, j, result1, result2)

    results1 = np.zeros(c.shape)
    results2 = np.zeros(c.shape)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_integrals, i, j) for i in range(c.shape[0]) for j in range(c.shape[1])]
        for future in futures:
            i, j, result1, result2 = future.result()
            results1[i, j] = result1
            results2[i, j] = result2

    return results1, results2

def check_sol(sigma, c, m, target1, target2):
    # GPU上のテンソルをCPUに移動
    sigma = sigma.cpu().numpy() if isinstance(sigma, torch.Tensor) else sigma
    c = c.cpu().numpy() if isinstance(c, torch.Tensor) else c
    m = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
    target1 = target1.cpu().numpy() if isinstance(target1, torch.Tensor) else target1
    target2 = target2.cpu().numpy() if isinstance(target2, torch.Tensor) else target2

    result1, result2 = integrate(sigma, c, m)
    #print("target1\n",target1)
    #print("result1\n",result1)
    #print("target2\n",target2)
    #print("result2\n",result2)
    return result1, result2


def loss_function_sol(params, x_t, sigma, target1, target2):
    m = params[:x_t.size].reshape(x_t.shape)
    c = params[x_t.size:].reshape(x_t.shape)

    result1, result2 = integrate(sigma, c, m)

    # target1 と target2 を numpy array に変換
    target1_np = target1.cpu().detach().numpy()
    target2_np = target2.cpu().detach().numpy()

    loss = ((result1 - target1_np) ** 2 + (result2 - target2_np) ** 2).sum()
    print(loss)

    return loss