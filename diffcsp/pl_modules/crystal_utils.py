import os
import numpy as np
import torch

# クローマー・マン係数の定義
cromer_mann_coefficients = {
    27: {'a': [15.7924, 6.1253, 3.28719, 1.64550], 'b': [2.77200, 0.90200, 0.21700, 9.25200], 'c': 1.79131},
    81: {'a': [29.2024, 15.1492, 14.5606, 5.98054], 'b': [1.14430, 10.0593, 0.21100, 27.0701], 'c': 13.4307},
    7:  {'a': [12.2126, 3.13220, 2.01250, 1.16630], 'b': [0.00570, 9.89330, 28.9975, 0.58260], 'c': -11.529},
    8:  {'a': [3.0485, 2.2868, 1.5463, 0.8670], 'b': [13.2771, 5.7011, 0.3239, 32.9089], 'c': 0.2508}
}

def initialize_coefficients(device='cuda:0'):
    """
    クローマー・マン係数を PyTorch テンソルとして初期化する関数。
    
    Parameters:
    - cromer_mann_coefficients (dict): 原子番号をキーとし、係数を含む辞書。
    - device (str): テンソルを配置するデバイス。
    
    Returns:
    - a_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - b_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - c_coeff (torch.Tensor): 形状 (max_atom_type+1,) のテンソル。
    """
    max_atom_type = max(cromer_mann_coefficients.keys())  # 最大原子番号を取得
    a_coeff = torch.zeros((max_atom_type+1, 4), device=device, dtype=torch.float32)  # 形状 (82, 4)
    b_coeff = torch.zeros((max_atom_type+1, 4), device=device, dtype=torch.float32)  # 形状 (82, 4)
    c_coeff = torch.zeros((max_atom_type+1), device=device, dtype=torch.float32)     # 形状 (82,)
    
    for atom, coeff in cromer_mann_coefficients.items():
        a_coeff[atom] = torch.tensor(coeff['a'], device=device, dtype=torch.float32)
        b_coeff[atom] = torch.tensor(coeff['b'], device=device, dtype=torch.float32)
        c_coeff[atom] = torch.tensor(coeff['c'], device=device, dtype=torch.float32)
    
    return a_coeff, b_coeff, c_coeff


def scattering_factor_torch(atom_types, q_magnitude, a_coeff, b_coeff, c_coeff):
    """
    PyTorchを使用して散乱因子を計算する関数。
    
    Parameters:
    - atom_types (torch.Tensor): 形状 (n_atoms,) のテンソル。各原子の原子番号。
    - q_magnitude (torch.Tensor): 形状 (num_k,) のテンソル。各kベクトルのqの大きさ。
    - a_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - b_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - c_coeff (torch.Tensor): 形状 (max_atom_type+1,) のテンソル。
    
    Returns:
    - f_q (torch.Tensor): 形状 (n_atoms, num_k) のテンソル。各原子の散乱因子。
    """
    pi = torch.pi
    a = a_coeff[atom_types]  # shape (n_atoms, 4)
    b = b_coeff[atom_types]  # shape (n_atoms, 4)
    c = c_coeff[atom_types]  # shape (n_atoms,)
    
    # Reshape for broadcasting
    a = a.unsqueeze(-1)  # (n_atoms, 4, 1)
    b = b.unsqueeze(-1)  # (n_atoms, 4, 1)
    q = q_magnitude.unsqueeze(0).unsqueeze(0)  # (1, 1, num_k)
    
    # Compute the exponent component
    exp_component = torch.exp(-b * (q / (4 * pi))**2)  # (n_atoms, 4, num_k)

    # Compute f_q: sum over j=1 to 4 of a_j * exp_component + c
    f_q = torch.sum(a * exp_component, dim=1) + c.unsqueeze(1)  # (n_atoms, num_k)
    
    return f_q  # shape (n_atoms, num_k)

def scattering_factor_torch_batch(atom_types, q_magnitude, a_coeff, b_coeff, c_coeff):
    """
    PyTorchを使用して散乱因子を計算する関数（バッチ対応）。
    
    Parameters:
    - atom_types (torch.Tensor): 形状 (C, A_max) のテンソル。各原子の原子番号。
    - q_magnitude (torch.Tensor): 形状 (num_k,) のテンソル。各kベクトルのqの大きさ。
    - a_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - b_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - c_coeff (torch.Tensor): 形状 (max_atom_type+1,) のテンソル。
    
    Returns:
    - f_q (torch.Tensor): 形状 (C, A_max, num_k) のテンソル。各原子の散乱因子。
    """
    pi = torch.pi
    C, A_max = atom_types.shape  # バッチのサイズと最大の原子数を取得
    
    # a, b, c の各係数を atom_types から選択
    a = a_coeff[atom_types]  # shape (C, A_max, 4)
    b = b_coeff[atom_types]  # shape (C, A_max, 4)
    c = c_coeff[atom_types]  # shape (C, A_max)
    
    # Reshape for broadcasting
    a = a.unsqueeze(-1)  # (C, A_max, 4, 1)
    b = b.unsqueeze(-1)  # (C, A_max, 4, 1)
    q = q_magnitude.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, num_k)
    
    # Compute the exponent component
    exp_component = torch.exp(-b * (q / (4 * pi))**2)  # (C, A_max, 4, num_k)
    
    # Compute f_q: sum over j=1 to 4 of a_j * exp_component + c
    f_q = torch.sum(a * exp_component, dim=2) + c.unsqueeze(-1)  # (C, A_max, num_k)
    
    return f_q  # shape (C, A_max, num_k)


def complex_sum_squared_with_scattering_factors_torch(
    k_grid, A_m, A_c, atom_types, a_coeff, b_coeff, c_coeff
):
    """
    PyTorchを使用して回折強度 I(hkl) を計算する関数。

    Parameters:
    - k_grid (torch.Tensor): 形状 (num_k, 3) のテンソル。kベクトルのグリッド。
    - A_m (torch.Tensor): 形状 (n_atoms, 3) のテンソル。原子の分率座標。
    - A_c (torch.Tensor): 形状 (n_atoms, 3) のテンソル。補正用の座標。
    - atom_types (torch.Tensor): 形状 (n_atoms,) のテンソル。各原子の原子番号。
    - a_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - b_coeff (torch.Tensor): 形状 (max_atom_type+1, 4) のテンソル。
    - c_coeff (torch.Tensor): 形状 (max_atom_type+1,) のテンソル。

    Returns:
    - real_I_hkl (torch.Tensor): 形状 (num_k,) のテンソル。各kベクトルに対するI(hkl)の実部。
    """
    pi = torch.pi

    # デバイスとデータ型の統一
    device = k_grid.device
    A_m = A_m.to(device).float()
    A_c = A_c.to(device).float()
    atom_types = atom_types.to(device)
    a_coeff = a_coeff.to(device).float()
    b_coeff = b_coeff.to(device).float()
    c_coeff = c_coeff.to(device).float()

    # qの計算
    q_magnitude = (2 * pi) * torch.norm(k_grid, dim=1)  # shape (num_k,)

    # 散乱因子の計算
    f_q = scattering_factor_torch(atom_types, q_magnitude, a_coeff, b_coeff, c_coeff)  # shape (n_atoms, num_k)

    # diff と k の計算
    diff = A_m.unsqueeze(1) - A_m.unsqueeze(0)  # shape (n_atoms, n_atoms, 3)

    # r_m の計算
    r_m = torch.einsum('nij,kj->nik', diff, k_grid)  # shape (num_k, n_atoms, n_atoms)

    # sum_c と k^2 の計算
    sum_c = A_c.unsqueeze(1) + A_c.unsqueeze(0)  # shape (n_atoms, n_atoms, 3)
    r_c = torch.einsum('nij,kj->nik', sum_c, k_grid**2)  # shape (num_k, n_atoms, n_atoms)

    # 散乱因子の積
    f_j = f_q.unsqueeze(1)  # shape (n_atoms, 1, num_k)
    f_m = f_q.unsqueeze(0)  # shape (1, n_atoms, num_k)
    f_jm = f_j * f_m  # shape (n_atoms, n_atoms, num_k)
    f_jm = f_jm.permute(2, 0, 1)  # shape (num_k, n_atoms, n_atoms)

    # 複素数演算の準備
    exponent = 2 * pi * 1j * r_m - 2 * (pi ** 2) * r_c  # shape (num_k, n_atoms, n_atoms)

    # 指数部の計算
    exp_term = torch.exp(exponent)  # shape (num_k, n_atoms, n_atoms)

    # exp_term の形状が正しいか確認
    if exp_term.shape != f_jm.shape:
        exp_term = exp_term.permute(2, 0, 1)  # shape (num_k, n_atoms, n_atoms) に揃える

    # デバッグ出力
    #print(f"f_jm shape: {f_jm.shape}, device: {f_jm.device}, dtype: {f_jm.dtype}")
    #print(f"exp_term shape: {exp_term.shape}, device: {exp_term.device}, dtype: {exp_term.dtype}")

    # I(hkl) の計算
    I_hkl = torch.sum(f_jm * exp_term, dim=(1, 2))  # shape (num_k,)

    # 実部のみを返す
    real_I_hkl = I_hkl.real / 10  # shape (num_k,)

    # デバッグ出力
    #print(f"I_hkl shape: {I_hkl.shape}, device: {I_hkl.device}, dtype: {I_hkl.dtype}")
    #print(f"real_I_hkl shape: {real_I_hkl.shape}, device: {real_I_hkl.device}, dtype: {real_I_hkl.dtype}")

    return real_I_hkl


def calculate_delI_delm_delx_t_vectorized(batch, m, c, delm_delx_t):
    """
    バッチ内の全ての結晶に対して構造因子のx_t微分を計算する関数。
    
    Args:
        batch (dict): バッチ情報を含む辞書。必要なキーは 'num_atoms' と 'atom_types'。
        m (torch.Tensor): m 座標テンソル [Total_atoms, 3]
        c (torch.Tensor): c 座標テンソル [Total_atoms, 3]
        delm_delx_t (torch.Tensor): delm/delx_t テンソル [Total_atoms]
    
    Returns:
        np.ndarray: 計算された Z テンソル [C, 5, 5, 5, num_atoms_max, 3]
    """
    device = batch['frac_coords'].device if 'frac_coords' in batch else 'cpu'
    num_atoms = batch['num_atoms'].to('cpu')  # [C]
    num_crystals = num_atoms.size(0)
    num_atoms_max = torch.max(num_atoms).item()
    m = m.to(device)  # [Total_atoms, 3]
    c = c.to(device)  # [Total_atoms, 3]
    # numpy配列である場合、torch.Tensorに変換
    if isinstance(delm_delx_t, np.ndarray):
        delm_delx_t = torch.tensor(delm_delx_t, dtype=torch.float32)
    delm_delx_t = delm_delx_t.to(device)  # [Total_atoms]
    atom_types = batch['atom_types'].to(device)  # [Total_atoms]
    
    # Kベクトルの生成
    k_range = torch.arange(-2, 3, device=device, dtype=m.dtype)
    K1, K2, K3 = torch.meshgrid(k_range, k_range, k_range, indexing='ij')  # 各 [5,5,5]
    K = torch.stack([K1, K2, K3], dim=-1).reshape(-1, 3)  # [125, 3]
    num_k = K.shape[0]
    
    # 出力テンソルの初期化
    Z = torch.zeros((num_crystals, num_k, num_atoms_max, 3), dtype=torch.float32, device=device)
    
    # クローマー・マン係数の初期化
    a_coeff, b_coeff, c_coeff = initialize_coefficients(device=device)
    
    # 各結晶のデータをパディングしてバッチ化
    padded_m = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_c = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_delm = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_atom_types = torch.zeros((num_crystals, num_atoms_max), dtype=torch.long, device=device)

    #print(m.shape)
    #print(delm_delx_t.shape)
    
    start = 0
    for i in range(num_crystals):
        n = num_atoms[i].item()
        padded_m[i, :n] = m[start:start+n]
        padded_c[i, :n] = c[start:start+n]
        padded_delm[i, :n] = delm_delx_t[start:start+n]
        padded_atom_types[i, :n] = atom_types[start:start+n]
        start += n
    
    # q の計算
    q = (2 * torch.pi) * torch.norm(K, dim=-1)  # [125]
    
    # 散乱因子の計算
    # f_i: [C, A_max, K]
    f_i = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]
    f_j = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]
    #(f_i.shape)
    
    # ペアワイズの差分と和の計算
    # diff_m: [C, A_max, A_max, 3]
    diff_m = padded_m.unsqueeze(2) - padded_m.unsqueeze(1)  # [C, A_max, A_max, 3]
    #print(diff_m.shape)
    sum_c = padded_c.unsqueeze(2) + padded_c.unsqueeze(1)    # [C, A_max, A_max, 3]
    #print(sum_c.shape)
    
    # Kベクトルを展開
    # K: [K, 3] -> [1, 1, 1, K, 3]
    K_expanded = K.view(1, 1, 1, num_k, 3)
    #print(K.shape)

    diff_m = diff_m.to(torch.double)
    K = K.to(torch.double)
    
    # r_m: [C, A_max, A_max, K] = dot(diff_m, K)
    r_m = torch.einsum('...ij,kj->...ik', diff_m, K)
    #print(r_m.shape)
    
    # r_c: [C, A_max, A_max, K] = dot(sum_c, K^2)
    K_sq = K ** 2  # [K, 3]
    sum_c = sum_c.to(torch.double)
    K_sq = K_sq.to(torch.double)
    r_c = torch.einsum('...ij,kj->...ik', sum_c, K_sq)  # [C, A_max, A_max, K]
    
    # 散乱因子の積
    # f_i: [C, A_max, K], f_j: [C, A_max, K]
    f_i_expand = f_i.unsqueeze(2)  # [C, A_max, 1, K]
    f_j_expand = f_j.unsqueeze(1)  # [C, 1, A_max, K]
    f_ij = f_i_expand * f_j_expand  # [C, A_max, A_max, K]
    
    # temp_result の計算
    # -4πK: [K, 3]
    temp = -4 * torch.pi * K  # [K, 3]
    # sin と exp の計算
    sin_term = torch.sin(2 * torch.pi * r_m)  # [C, A_max, A_max, K]
    exp_term = torch.exp(-2 * (torch.pi ** 2) * r_c)  # [C, A_max, A_max, K]
    
    # delm_coords[i]: [C, A_max, 1, 1]
    delm_i = padded_delm.unsqueeze(2) # [C, A_max, 1, 1]
    #(delm_i.shape)

    
    # temp_result: [C, A_max, A_max, K, 3]
    temp_result = (temp.unsqueeze(0).unsqueeze(0) * f_ij.unsqueeze(-1) * sin_term.unsqueeze(-1) * exp_term.unsqueeze(-1))  # [C, A_max, A_max, K, 3]

    # delm_coords[i] を乗算
    #print(temp_result.shape)
    # delm_i の形状を [16, 5, 5, 125, 3] に揃える
    delm_i = delm_i.unsqueeze(3)  # [16, 5, 1, 1, 3]
    #print(delm_i.shape)
    delm_i = delm_i.expand(-1, -1, 5, 125, -1)  # [16, 5, 5, 125, 3]
    #print(delm_i.shape)
    temp_result = temp_result * delm_i  # [C, A_max, A_max, K, 3]
    
    # j に対して合計
    sum_j = torch.sum(temp_result, dim=2)  # [C, A_max, K, 3]
    sum_j = sum_j.permute(0, 2, 1, 3)  # [C, K, A_max, 3]
    #print(sum_j.shape)
    
    # Z に格納
    Z += sum_j  # [C, K, A_max, 3]

    Z = Z.view(num_crystals, 5, 5, 5, num_atoms_max, 3)
    
    # 最後にZをnumpy.ndarrayに変換
    Z_numpy = Z.cpu().numpy()
    
    return Z_numpy


def calculate_y_vectorized_re(batch, c):
    """
    バッチ内の全ての結晶に対して構造因子のx_t微分を計算する関数。
    
    Args:
        batch (dict): バッチ情報を含む辞書。必要なキーは 'num_atoms' と 'atom_types'。
        m (torch.Tensor): m 座標テンソル [Total_atoms, 3]
        c (torch.Tensor): c 座標テンソル [Total_atoms, 3]
        delm_delx_t (torch.Tensor): delm/delx_t テンソル [Total_atoms]
    
    Returns:
        np.ndarray: 計算された Z テンソル [C, 5, 5, 5, num_atoms_max, 3]
    """
    device = batch['frac_coords'].device if 'frac_coords' in batch else 'cpu'
    num_atoms = batch['num_atoms'].to('cpu')  # [C]
    num_crystals = num_atoms.size(0)
    num_atoms_max = torch.max(num_atoms).item()
    m = batch['frac_coords']
    c = c.to(device)  # [Total_atoms, 3]
    atom_types = batch['atom_types'].to(device)  # [Total_atoms]
    
    # Kベクトルの生成
    k_range = torch.arange(-2, 3, device=device, dtype=m.dtype)
    K1, K2, K3 = torch.meshgrid(k_range, k_range, k_range, indexing='ij')  # 各 [5,5,5]
    K = torch.stack([K1, K2, K3], dim=-1).reshape(-1, 3)  # [125, 3]
    num_k = K.shape[0]
    
    # 出力テンソルの初期化
    Z = torch.zeros((num_crystals, num_k), dtype=torch.float32, device=device)
    
    # クローマー・マン係数の初期化
    a_coeff, b_coeff, c_coeff = initialize_coefficients(device=device)
    
    # 各結晶のデータをパディングしてバッチ化
    padded_m = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_c = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_atom_types = torch.zeros((num_crystals, num_atoms_max), dtype=torch.long, device=device)
    
    start = 0
    for i in range(num_crystals):
        n = num_atoms[i].item()
        padded_m[i, :n] = m[start:start+n]
        padded_c[i, :n] = c[start:start+n]
        padded_atom_types[i, :n] = atom_types[start:start+n]
        start += n
    
    # q の計算
    q = (2 * torch.pi) * torch.norm(K, dim=-1)  # [125]
    
    # 散乱因子の計算
    # f_i: [C, A_max, K]
    f_i = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]
    f_j = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]
    #(f_i.shape)
    
    # ペアワイズの差分と和の計算
    # diff_m: [C, A_max, A_max, 3]
    diff_m = padded_m.unsqueeze(2) - padded_m.unsqueeze(1)  # [C, A_max, A_max, 3]
    #print(diff_m.shape)
    sum_c = padded_c.unsqueeze(2) + padded_c.unsqueeze(1)    # [C, A_max, A_max, 3]
    #print(sum_c.shape)
    
    # Kベクトルを展開
    # K: [K, 3] -> [1, 1, 1, K, 3]
    K_expanded = K.view(1, 1, 1, num_k, 3)
    #print(K.shape)

    diff_m = diff_m.to(torch.double)
    K = K.to(torch.double)
    
    # r_m: [C, A_max, A_max, K] = dot(diff_m, K)
    r_m = torch.einsum('...ij,kj->...ik', diff_m, K)
    #print(r_m.shape)
    
    # r_c: [C, A_max, A_max, K] = dot(sum_c, K^2)
    K_sq = K ** 2  # [K, 3]
    sum_c = sum_c.to(torch.double)
    K_sq = K_sq.to(torch.double)
    r_c = torch.einsum('...ij,kj->...ik', sum_c, K_sq)  # [C, A_max, A_max, K]
    
    # 散乱因子の積
    # f_i: [C, A_max, K], f_j: [C, A_max, K]
    f_i_expand = f_i.unsqueeze(2)  # [C, A_max, 1, K]
    f_j_expand = f_j.unsqueeze(1)  # [C, 1, A_max, K]
    f_ij = f_i_expand * f_j_expand  # [C, A_max, A_max, K]
    
    # sin と exp の計算
    exp_term = torch.exp(2 * torch.pi * 1j * r_m - 2 * (torch.pi ** 2) * r_c)  # [C, A_max, A_max, K]
    
    # temp_result: [C, A_max, A_max, K]
    temp_result = (f_ij * exp_term)  # [C, A_max, A_max, K]
    # print(temp_result.shape)
    
    # j に対して合計
    sum_j = torch.sum(temp_result, dim=2)  # [C, A_max, K]
    # print(sum_j.shape)
    sum_j = torch.sum(sum_j, dim=1)
    # print(sum_j.shape)

    sum_j = sum_j.real / 10
    
    # Z に格納
    Z += sum_j  # [C, K]
    Z = Z.view(num_crystals, 5, 5, 5)
    
    # 最後にZをnumpy.ndarrayに変換
    Z_numpy = Z.cpu().numpy()
    
    return Z_numpy

def calculate_I_vectorized_re(batch, m, c):
    """
    バッチ内の全ての結晶に対して構造因子のx_t微分を計算する関数。
    
    Args:
        batch (dict): バッチ情報を含む辞書。必要なキーは 'num_atoms' と 'atom_types'。
        m (torch.Tensor): m 座標テンソル [Total_atoms, 3]
        c (torch.Tensor): c 座標テンソル [Total_atoms, 3]
        delm_delx_t (torch.Tensor): delm/delx_t テンソル [Total_atoms]
    
    Returns:
        np.ndarray: 計算された Z テンソル [C, 5, 5, 5, num_atoms_max, 3]
    """
    device = batch['frac_coords'].device if 'frac_coords' in batch else 'cpu'
    num_atoms = batch['num_atoms'].to('cpu')  # [C]
    num_crystals = num_atoms.size(0)
    num_atoms_max = torch.max(num_atoms).item()
    m = m.to(device)
    c = c.to(device)  # [Total_atoms, 3]
    atom_types = batch['atom_types'].to(device)  # [Total_atoms]
    
    # Kベクトルの生成
    k_range = torch.arange(-2, 3, device=device, dtype=m.dtype)
    K1, K2, K3 = torch.meshgrid(k_range, k_range, k_range, indexing='ij')  # 各 [5,5,5]
    K = torch.stack([K1, K2, K3], dim=-1).reshape(-1, 3)  # [125, 3]
    num_k = K.shape[0]
    
    # 出力テンソルの初期化
    Z = torch.zeros((num_crystals, num_k), dtype=torch.float32, device=device)
    
    # クローマー・マン係数の初期化
    a_coeff, b_coeff, c_coeff = initialize_coefficients(device=device)
    
    # 各結晶のデータをパディングしてバッチ化
    padded_m = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_c = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_atom_types = torch.zeros((num_crystals, num_atoms_max), dtype=torch.long, device=device)
    
    start = 0
    for i in range(num_crystals):
        n = num_atoms[i].item()
        padded_m[i, :n] = m[start:start+n]
        padded_c[i, :n] = c[start:start+n]
        padded_atom_types[i, :n] = atom_types[start:start+n]
        start += n
    
    # q の計算
    q = (2 * torch.pi) * torch.norm(K, dim=-1)  # [125]
    
    # 散乱因子の計算
    # f_i: [C, A_max, K]
    f_i = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]
    f_j = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]
    #(f_i.shape)
    
    # ペアワイズの差分と和の計算
    # diff_m: [C, A_max, A_max, 3]
    diff_m = padded_m.unsqueeze(2) - padded_m.unsqueeze(1)  # [C, A_max, A_max, 3]
    #print(diff_m.shape)
    sum_c = padded_c.unsqueeze(2) + padded_c.unsqueeze(1)    # [C, A_max, A_max, 3]
    #print(sum_c.shape)
    
    # Kベクトルを展開
    # K: [K, 3] -> [1, 1, 1, K, 3]
    K_expanded = K.view(1, 1, 1, num_k, 3)
    #print(K.shape)

    diff_m = diff_m.to(torch.double)
    K = K.to(torch.double)
    
    # r_m: [C, A_max, A_max, K] = dot(diff_m, K)
    r_m = torch.einsum('...ij,kj->...ik', diff_m, K)
    #print(r_m.shape)
    
    # r_c: [C, A_max, A_max, K] = dot(sum_c, K^2)
    K_sq = K ** 2  # [K, 3]
    sum_c = sum_c.to(torch.double)
    K_sq = K_sq.to(torch.double)
    r_c = torch.einsum('...ij,kj->...ik', sum_c, K_sq)  # [C, A_max, A_max, K]
    
    # 散乱因子の積
    # f_i: [C, A_max, K], f_j: [C, A_max, K]
    f_i_expand = f_i.unsqueeze(2)  # [C, A_max, 1, K]
    f_j_expand = f_j.unsqueeze(1)  # [C, 1, A_max, K]
    f_ij = f_i_expand * f_j_expand  # [C, A_max, A_max, K]
    
    # sin と exp の計算
    exp_term = torch.exp(2 * torch.pi * 1j * r_m - 2 * (torch.pi ** 2) * r_c)  # [C, A_max, A_max, K]
    
    # temp_result: [C, A_max, A_max, K]
    temp_result = (f_ij * exp_term)  # [C, A_max, A_max, K]
    # print(temp_result.shape)
    
    # j に対して合計
    sum_j = torch.sum(temp_result, dim=2)  # [C, A_max, K]
    # print(sum_j.shape)
    sum_j = torch.sum(sum_j, dim=1)
    # print(sum_j.shape)

    sum_j = sum_j.real / 10
    
    # Z に格納
    Z += sum_j  # [C, K]
    Z = Z.view(num_crystals, 5, 5, 5)
    
    # 最後にZをnumpy.ndarrayに変換
    Z_numpy = Z.cpu().numpy()
    
    return Z_numpy

def calculate_delI_delc_delx_t_vectorized(batch, m, c, delc_delx_t):
    """
    バッチ内の全ての結晶に対して構造因子のc_t微分を計算する関数。
    
    Args:
        batch (dict): バッチ情報を含む辞書。必要なキーは 'num_atoms' と 'atom_types'。
        m (torch.Tensor): m 座標テンソル [Total_atoms, 3]
        c (torch.Tensor): c 座標テンソル [Total_atoms, 3]
        delc_delx_t (torch.Tensor): delc/delx_t テンソル [Total_atoms]
    
    Returns:
        np.ndarray: 計算された Z テンソル [C, 5, 5, 5, num_atoms_max, 3]
    """
    device = batch['frac_coords'].device if 'frac_coords' in batch else 'cpu'
    num_atoms = batch['num_atoms'].to('cpu')  # [C]
    num_crystals = num_atoms.size(0)
    num_atoms_max = torch.max(num_atoms).item()
    m = m.to(device)  # [Total_atoms, 3]
    c = c.to(device)  # [Total_atoms, 3]
    if isinstance(delc_delx_t, np.ndarray):
        delc_delx_t = torch.tensor(delc_delx_t, dtype=torch.float32)
    delc_delx_t = delc_delx_t.to(device)  # [Total_atoms]
    atom_types = batch['atom_types'].to(device)  # [Total_atoms]

    # Kベクトルの生成
    k_range = torch.arange(-2, 3, device=device, dtype=m.dtype)
    K1, K2, K3 = torch.meshgrid(k_range, k_range, k_range, indexing='ij')  # 各 [5,5,5]
    K = torch.stack([K1, K2, K3], dim=-1).reshape(-1, 3)  # [125, 3]
    num_k = K.shape[0]
    
    # 出力テンソルの初期化
    Z = torch.zeros((num_crystals, num_k, num_atoms_max, 3), dtype=torch.float32, device=device)
    
    # クローマー・マン係数の初期化
    a_coeff, b_coeff, c_coeff = initialize_coefficients(device=device)

    # 各結晶のデータをパディングしてバッチ化
    padded_m = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_c = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_delc = torch.zeros((num_crystals, num_atoms_max, 3), dtype=torch.float32, device=device)
    padded_atom_types = torch.zeros((num_crystals, num_atoms_max), dtype=torch.long, device=device)

    start = 0
    for i in range(num_crystals):
        n = num_atoms[i].item()
        padded_m[i, :n] = m[start:start+n]
        padded_c[i, :n] = c[start:start+n]
        padded_delc[i, :n] = delc_delx_t[start:start+n]
        padded_atom_types[i, :n] = atom_types[start:start+n]
        start += n

    # q の計算
    q = (2 * torch.pi) * torch.norm(K, dim=-1)  # [125]
    K_squared = K ** 2  # [125, 3]

    # 散乱因子の計算
    f_i = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]
    f_j = scattering_factor_torch_batch(padded_atom_types, q, a_coeff, b_coeff, c_coeff)  # [C, A_max, K]

    # ペアワイズの差分と和の計算
    diff_m = padded_m.unsqueeze(2) - padded_m.unsqueeze(1)  # [C, A_max, A_max, 3]
    sum_c = padded_c.unsqueeze(2) + padded_c.unsqueeze(1)    # [C, A_max, A_max, 3]
    
    # r_m: [C, A_max, A_max, K] = dot(diff_m, K)
    diff_m = diff_m.to(torch.double)
    K = K.to(torch.double)
    r_m = torch.einsum('...ij,kj->...ik', diff_m, K)  # [C, A_max, A_max, K]
    
    # r_c: [C, A_max, A_max, K] = dot(sum_c, K_squared)
    sum_c = sum_c.to(torch.double)
    K_squared = K_squared.to(torch.double)
    r_c = torch.einsum('...ij,kj->...ik', sum_c, K_squared)  # [C, A_max, A_max, K]
    
    # 散乱因子の積
    f_i_expand = f_i.unsqueeze(2)  # [C, A_max, 1, K]
    f_j_expand = f_j.unsqueeze(1)  # [C, 1, A_max, K]
    f_ij = f_i_expand * f_j_expand  # [C, A_max, A_max, K]
    
    # temp_result の計算
    # -4 * np.pi**2 * K_squared * np.cos(2 * np.pi * r_m) * np.exp(-2 * np.pi**2 * r_c)
    temp = -4 * (torch.pi ** 2) * K_squared  # [K, 3]
    cos_term = torch.cos(2 * torch.pi * r_m)  # [C, A_max, A_max, K]
    exp_term = torch.exp(-2 * (torch.pi ** 2) * r_c)  # [C, A_max, A_max, K]
    
    # delc_coords[i]: [C, A_max, 1, 1]
    delc_i = padded_delc.unsqueeze(2)  # [C, A_max, 1, 3]
    
    # temp_result: [C, A_max, A_max, K, 3]
    temp_result = (temp.unsqueeze(0).unsqueeze(0) * f_ij.unsqueeze(-1) * cos_term.unsqueeze(-1) * exp_term.unsqueeze(-1))  # [C, A_max, A_max, K, 3]

    # delc_coords[i] を乗算
    delc_i = delc_i.unsqueeze(3).expand(-1, -1, 5, 125, -1)  # [C, A_max, 5, 125, 3]
    temp_result = temp_result * delc_i  # [C, A_max, A_max, K, 3]
    
    # j に対して合計
    sum_j = torch.sum(temp_result, dim=2)  # [C, A_max, K, 3]
    sum_j = sum_j.permute(0, 2, 1, 3)  # [C, K, A_max, 3]
    
    # Z に格納
    Z += sum_j  # [C, K, A_max, 3]

    # 125を5x5x5に再構成
    Z = Z.view(num_crystals, 5, 5, 5, num_atoms_max, 3)
    
    # 最後にZをnumpy.ndarrayに変換
    Z_numpy = Z.cpu().numpy()
    
    return Z_numpy


"""
======================
        Archive
======================
"""

def calculate_q_magnitude(k_vector, lambda_wavelength=1.0):
    # Ensure k_vector is a tuple
    if isinstance(k_vector, list):
        k_vector = tuple(k_vector)
    
    # Calculate the magnitude of the q vector from the k vector
    return (2 * np.pi / lambda_wavelength) * np.linalg.norm(k_vector)

def scattering_factor(atom_number, q):
    
    coefficients = cromer_mann_coefficients.get(atom_number.item())
    if not coefficients:
        raise ValueError(f"Atomic number {atom_number} not supported.")
    
    a = coefficients['a']
    b = coefficients['b']
    c = coefficients['c']
    
    f_q = sum([a[i] * np.exp(-b[i] * (q / (4 * np.pi)) ** 2) for i in range(4)]) + c
    # print(atom_number, f_q)
    return f_q

def complex_sum_squared_with_scattering_factors(k, A_m, A_c, atom_types):
    """
    3次元ベクトル k と (n x 3) の行列 A、および散乱因子のリスト f を受け取り、
    I(hkl) = |F(hkl)|^2 を計算する関数。
    F(hkl) = sum_j f_j * exp(2 * pi * i * (hx_j + ky_j + lz_j))
    I(hkl) = sum_j sum_k f_j * f_k * exp(2 * pi * i * ((x_j - x_k)h + (y_j - y_k)k + (z_j - z_k)l))

    Parameters:
    k (np.ndarray): 3次元ベクトル (h, k, l)
    A (np.ndarray): (n x 3) の行列 (原子の分率座標)
    f (np.ndarray): (n) の配列 (原子の散乱因子)

    Returns:
    float: 回折強度 I(hkl)
    """
    i = complex(0, 1)
    pi = np.pi

    # CUDAテンソルをCPUに移動させてNumPy配列に変換
    if isinstance(A_m, torch.Tensor):
        A_m = A_m.cpu().numpy()
    if isinstance(A_c, torch.Tensor):
        A_c= A_c.cpu().numpy()

    # 行数を取得
    n = A_m.shape[0]

    # 回折強度 I(hkl) の計算
    result = 0.0
    for j in range(n):
        for m in range(n):
            f_j = scattering_factor(atom_types[j], calculate_q_magnitude(k))
            f_m = scattering_factor(atom_types[m], calculate_q_magnitude(k))
            diff = A_m[j] - A_m[m]
            sum_c = A_c[j] + A_c[m]
            r_m = np.dot(diff, k)
            r_c = np.dot(sum_c, k**2)
            result += f_j * f_m * np.exp(2 * pi * i * r_m - 2 * pi**2 * r_c)

    # 結果の実部のみを返す
    real_result = np.real(result)
    # print(real_result)
    return real_result / 10