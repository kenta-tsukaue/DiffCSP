import os
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.vis.structure_vtk import StructureVis  # VTKベースの可視化
import matplotlib.pyplot as plt
from collections import defaultdict

from get_af0 import af0

def visualize_structure_with_matplotlib(structure):
    """
    結晶構造をmatplotlibで可視化する関数
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 原子種ごとに座標をまとめる
    atom_coords = defaultdict(list)
    for site in structure:
        atom_coords[site.species_string].append(site.frac_coords)
    
    # 色のループを作成
    colors = plt.cm.get_cmap('tab20', len(atom_coords))  # tab20カラーマップを使用
    for idx, (atom_type, coords) in enumerate(atom_coords.items()):
        coords = np.array(coords)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                   s=100, color=colors(idx), label=atom_type, alpha=0.6)

    # 座標軸の範囲を0から1に固定
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # 座標軸のラベル設定
    ax.set_xlabel('X Fractional Coordinate')
    ax.set_ylabel('Y Fractional Coordinate')
    ax.set_zlabel('Z Fractional Coordinate')
    ax.set_title('Crystal Structure')
    ax.legend()  # 凡例を表示

    plt.show()

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
    print(atom_number, f_q)
    return f_q

def scattering_factor_with_table(atom_number, K, af0_table):
    h = K[0]
    k = K[1]
    l = K[2]
    
    # dfから条件に一致する行をフィルタリング
    filtered_row = af0_table[
        (af0_table['atom_num'] == atom_number.item()) &
        (af0_table['h'] == h) &
        (af0_table['k'] == k) &
        (af0_table['l'] == l)
    ]
    
    # フィルタされた行からaf0を取得し、f_qとして返す
    if not filtered_row.empty:
        f_q = filtered_row.iloc[0]['af0']  # 一致する行が複数あった場合、最初の行を使用
        return f_q
    else:
        print("エラー")

def complex_sum_squared_with_scattering_factors(k, A, atom_types):
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
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    # 行数を取得
    n = A.shape[0]

    # 回折強度 I(hkl) の計算
    result = 0.0
    for j in range(n):
        for m in range(n):
            f_j = scattering_factor(atom_types[j], calculate_q_magnitude(k))
            f_m = scattering_factor(atom_types[m], calculate_q_magnitude(k))
            diff = A[j] - A[m]
            r = np.dot(diff, k)
            result += f_j * f_m * np.exp(2 * pi * i * r)

    # 結果の実部のみを返す
    real_result = np.real(result)
    print(real_result)
    return real_result

def complex_sum_squared_with_scattering_factors_with_table(k, A, atom_types, af0_table):
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
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    # 行数を取得
    n = A.shape[0]

    # 回折強度 I(hkl) の計算
    result = 0.0
    for j in range(n):
        for m in range(n):
            f_j = scattering_factor_with_table(atom_types[j], k, af0_table)
            f_m = scattering_factor_with_table(atom_types[m], k, af0_table)
            diff = A[j] - A[m]
            r = np.dot(diff, k)
            result += f_j * f_m * np.exp(2 * pi * i * r)

    # 結果の実部のみを返す
    real_result = np.real(result)
    print(real_result)
    return real_result


def visualize_complex_sum(A, num_atoms, atom_types):
    # kの範囲設定
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    # 結果を格納する配列
    Z = np.zeros((len(k1_values), len(k2_values)))

    af0_table = af0()

    # k1とk2を動かしてcomplex_sumの値を計算
    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            k = np.array([0, k1, k2])
            #Z[i, j] = complex_sum_squared_with_scattering_factors(k, A, atom_types)
            Z[i, j] = complex_sum_squared_with_scattering_factors_with_table(k, A, atom_types, af0_table)

    print(Z)

    # プロット
    X, Y = np.meshgrid(k1_values, k2_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z.T, cmap='viridis')

    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    ax.set_zlabel(f'|complex_sum|')
    ax.set_title(f'3D Plot of complex_sum /TlCoN2O')
    plt.show()
    

def main():
    # データの呼び出し、データをCPUにマッピング
    loaded_batch = torch.load('sample/sample_d1_44_20/traj_200.pt', map_location=torch.device('cpu'))
    # 読み込んだデータを使用

    num_crystals = loaded_batch['num_atoms'].size(0)  # バッチサイズ

    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
    #for i in range(1, 2):
        start_index = sum(loaded_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        end_index = start_index + loaded_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        # i番目の結晶のデータを抽出
        first_frac_coords = loaded_batch['frac_coords'][start_index:end_index]
        first_atom_types = loaded_batch['atom_types'][start_index:end_index]
        num_atoms = loaded_batch['num_atoms'][i]
        lattice = loaded_batch['lattices'][i]
        m = loaded_batch['m'][start_index:end_index] + 0.5
        c = torch.full_like(m, 0.007)

        # pymatgenのStructureオブジェクトを作成
        structure = Structure(lattice, first_atom_types, first_frac_coords)

        # 結晶構造を可視化
        visualize_structure_with_matplotlib(structure)  # 可視化関数を呼び出し
        print(first_frac_coords)

        # Fを計算
        visualize_complex_sum(first_frac_coords, num_atoms, first_atom_types)


if __name__ == "__main__":
    main()
