#  ある自由度で、方程式の解(m,c)を数値積分でチェックする
import numpy as np
import torch
from scipy.integrate import quad
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import RegularGridInterpolator


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


def calculate_derivative_of_m(x_t, delta, generate_tables, pred_x, pred_x_d2, sigma_x, find_best_fit):
    # x_t を delta だけ増減させた値を計算
    x_t_plus = x_t + delta
    x_t_minus = x_t - delta

    # x_t のときのテーブルを生成
    s1_table_plus, s2_table_plus, m_values, c_values = generate_tables(x_t_plus)
    s1_table_minus, s2_table_minus, m_values, c_values = generate_tables(x_t_minus)

    # x_tが+-delta移動した際のpred_xの値を算出
    pred_x_plus = pred_x + (delta * pred_x_d2)
    pred_x_minus = pred_x - (delta * pred_x_d2)
    print(-1/sigma_x)
    print(pred_x_d2)

    # 補間関数を作成
    interp_s1_list_plus = []
    interp_s2_list_plus = []
    interp_s1_list_minus = []
    interp_s2_list_minus = []

    """n, d, _, _ = s1_table_plus.shape
    for i in range(n):
        for j in range(d):
            interp_s1_plus = RegularGridInterpolator((m_values, c_values), s1_table_plus[i, j, :, :], method='cubic', bounds_error=False, fill_value=None)
            interp_s2_plus = RegularGridInterpolator((m_values, c_values), s2_table_plus[i, j, :, :], method='cubic', bounds_error=False, fill_value=None)
            interp_s1_list_plus.append(interp_s1_plus)
            interp_s2_list_plus.append(interp_s2_plus)

            interp_s1_minus = RegularGridInterpolator((m_values, c_values), s1_table_minus[i, j, :, :], method='cubic', bounds_error=False, fill_value=None)
            interp_s2_minus = RegularGridInterpolator((m_values, c_values), s2_table_minus[i, j, :, :], method='cubic', bounds_error=False, fill_value=None)
            interp_s1_list_minus.append(interp_s1_minus)
            interp_s2_list_minus.append(interp_s2_minus)"""

    # x_t_plus のときの m を計算
    #m_plus, c_plus = find_best_fit_interp(interp_s1_list_plus, interp_s2_list_plus, x_t_plus, pred_x, pred_x_d2, sigma_x, m_values, c_values)
    m_plus, c_plus = find_best_fit(s1_table_plus, s2_table_plus, m_values, c_values, pred_x_plus, pred_x_d2, sigma_x)
    # x_t_minus のときの m を計算
    #m_minus, c_minus = find_best_fit_interp(interp_s1_list_minus, interp_s2_list_minus, x_t_minus, pred_x, pred_x_d2, sigma_x, m_values, c_values)
    m_minus, c_minus = find_best_fit(s1_table_minus, s2_table_minus, m_values, c_values, pred_x_minus, pred_x_d2, sigma_x)
    
    # デバッグ出力
    print("m_plus", m_plus)
    print("m_minus", m_minus)
    print("c_plus", c_plus)
    print("c_minus", c_minus)
    with open('output_m_c.txt', 'w') as f:
        # ファイルオブジェクトのwriteメソッドを使用して書き込む
        f.write(f"\nm_plus:\n{m_plus}")
        f.write(f"\nm_minus:\n{m_minus}")
        f.write(f"\nc_plus: \n{c_plus.shape, c_plus}")
        f.write(f"\nc_minus: \n{c_minus.shape, c_minus}")

    # m の微分を計算
    derivative_of_m = (m_plus - m_minus) / (2 * delta)
    derivative_of_c = (c_plus - c_minus) / (2 * delta)

    return derivative_of_m, derivative_of_c

def refine_grid(m_values, c_values, factor=3):
    m_min, m_max = np.min(m_values), np.max(m_values)
    c_min, c_max = np.min(c_values), np.max(c_values)
    
    m_values_fine = np.linspace(m_min, m_max, len(m_values) * factor)
    c_values_fine = np.linspace(c_min, c_max, len(c_values) * factor)
    
    return m_values_fine, c_values_fine


def find_best_fit_interp(interp_s1_list, interp_s2_list, x_t, score1, score2, sigma, m_values, c_values):
    x_t_cpu = x_t.cpu().detach().numpy()
    score1_cpu = score1.cpu().detach().numpy()
    score2_cpu = score2.cpu().detach().numpy()
    sigma_cpu = sigma.cpu().detach().numpy()
    n, d = x_t_cpu.shape
    m = np.zeros((n, d))
    c = np.zeros((n, d))

    # Refine the grid
    m_values_fine, c_values_fine = refine_grid(m_values, c_values)

    idx = 0
    for i in range(n):
        print(i)
        for j in range(d):
            interp_s1 = interp_s1_list[idx]
            interp_s2 = interp_s2_list[idx]
            # Interpolation requires 2D input, so we stack m_values_fine and c_values_fine
            grid_points = np.stack(np.meshgrid(m_values_fine, c_values_fine), axis=-1).reshape(-1, 2)
            s1 = interp_s1(grid_points)
            s2 = interp_s2(grid_points)
            
            # Reshape s1 and s2 to the shape of (len(m_values_fine), len(c_values_fine))
            s1 = s1.reshape(len(m_values_fine), len(c_values_fine))
            s2 = s2.reshape(len(m_values_fine), len(c_values_fine))
            
            # Compute error
            err = (sigma_cpu**2 * score1_cpu[i, j] - s1)**2 + (sigma_cpu**4 * score2_cpu[i, j] + sigma_cpu**2 - s2)**2
            
            # Find the indices of the minimum error
            m_idx, c_idx = np.unravel_index(np.argmin(err), err.shape)
            
            # Store the best m and c
            m[i, j] = m_values_fine[m_idx]
            c[i, j] = c_values_fine[c_idx]
            idx += 1

    return m, c



def calculate_partial_derivative_I_with_respect_to_m_and_c(batch, m, c, delta, calculate_I):
    num_atoms = batch.num_atoms
    I_shape = calculate_I(num_atoms, batch, m, c).shape
    m_shape = m.shape
    c_shape = c.shape
    
    # m に対する偏微分を格納するための配列を初期化
    partial_derivative_I_m = np.zeros(I_shape + m_shape)
    
    # c に対する偏微分を格納するための配列を初期化
    partial_derivative_I_c = np.zeros(I_shape + c_shape)
    
    # m の各要素に対して偏微分を計算
    for i in range(m_shape[0]):
        for j in range(m_shape[1]):
            # m のコピーを作成して微小変化を加える
            m_plus = m.copy()
            m_minus = m.copy()
            m_plus[i, j] += delta
            m_minus[i, j] -= delta
            
            # 微小変化を加えたときの I を計算
            I_plus = calculate_I(num_atoms, batch, m_plus, c)
            I_minus = calculate_I(num_atoms, batch, m_minus, c)
            
            # 偏微分を計算
            partial_derivative = (I_plus - I_minus) / (2 * delta)
            
            # 偏微分を結果配列に格納
            partial_derivative_I_m[..., i, j] = partial_derivative
    
    # c の各要素に対して偏微分を計算
    for i in range(c_shape[0]):
        for j in range(c_shape[1]):
            # c のコピーを作成して微小変化を加える
            c_plus = c.copy()
            c_minus = c.copy()
            c_plus[i, j] += delta
            c_minus[i, j] -= delta
            
            # 微小変化を加えたときの I を計算
            I_plus = calculate_I(num_atoms, batch, m, c_plus)
            I_minus = calculate_I(num_atoms, batch, m, c_minus)
            
            # 偏微分を計算
            partial_derivative = (I_plus - I_minus) / (2 * delta)
            
            # 偏微分を結果配列に格納
            partial_derivative_I_c[..., i, j] = partial_derivative
    
    return partial_derivative_I_m, partial_derivative_I_c