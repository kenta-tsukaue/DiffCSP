import os
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.vis.structure_vtk import StructureVis  # VTKベースの可視化
import matplotlib.pyplot as plt

def visualize_structure(structure):
    """結晶構造を可視化するヘルパー関数"""
    vis = StructureVis()
    vis.set_structure(structure)
    vis.show()

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

    # 各行とベクトル k の内積を計算
    r = np.dot(A, k)

    # exp(2 * pi * i * r) の和を計算
    result = np.sum(np.exp(2 * pi * i * r))

    return result

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

def I_hkl(k, A_m, A_c):
    """
    3次元ベクトル k と (n x 3) の行列 A_m, A_c を受け取り、
    I(h,k,l) = sum_j sum_k exp{2 * pi * I [(m_j - m_k)・(h,k,l) - 2 * pi^2 (c_j + c_k)・(h^2, k^2, l^2)]}
    を計算する関数。

    Parameters:
    k (np.ndarray): 3次元ベクトル (h, k, l)
    A_m (np.ndarray): (n x 3) の行列 m_j
    A_c (np.ndarray): (n x 3) の行列 c_j

    Returns:
    complex: 複素数の和
    """
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

def visualize_complex_sum(A, m, c, num_atoms):
    # kの範囲設定
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    # 結果を格納する配列
    Z = np.zeros((len(k1_values), len(k2_values)))

    # k1とk2を動かしてcomplex_sumの値を計算
    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            k = np.array([-2, k1, k2])
            # Z[i, j] = np.abs(complex_sum_squared(k, m))
            Z[i, j] = np.abs(complex_sum_squared(k, A))
            #Z[i, j] = np.abs(I_hkl(k, m, c))

    print(Z)

    # プロット
    X, Y = np.meshgrid(k1_values, k2_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z.T, cmap='viridis')

    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    ax.set_zlabel(f'|complex_sum|')
    ax.set_title(f'3D Plot of complex_sum /C{num_atoms}')
    plt.show()
    

def main():
    # データの呼び出し、データをCPUにマッピング
    loaded_batch = torch.load('sample/sample_d1_43_101/traj.pt', map_location=torch.device('cpu'))
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
        visualize_structure(structure)  # 可視化関数を呼び出し
        print(first_frac_coords)

        # Fを計算
        # visualize_complex_sum(first_frac_coords, m, c, num_atoms)


if __name__ == "__main__":
    main()
