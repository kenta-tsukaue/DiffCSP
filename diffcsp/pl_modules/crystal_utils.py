import os
import numpy as np
import torch

cromer_mann_coefficients = {
    27: {'a': [15.7924, 6.1253, 3.28719, 1.64550], 'b': [2.77200, 0.90200, 0.21700, 9.25200], 'c': 1.79131},
    81: {'a': [29.2024, 15.1492, 14.5606, 5.98054], 'b': [1.14430, 10.0593, 0.21100, 27.0701], 'c': 13.4307},
    7:  {'a': [12.2126, 3.13220, 2.01250, 1.16630], 'b': [0.00570, 9.89330, 28.9975, 0.58260], 'c': -11.529},
    8:  {'a': [3.0485, 2.2868, 1.5463, 0.8670], 'b': [13.2771, 5.7011, 0.3239, 32.9089], 'c': 0.2508}
}

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