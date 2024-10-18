from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import numpy as np
import random
import japanize_matplotlib
import matplotlib.pyplot as plt
import torch
from get_fq import get_fq
from collections import defaultdict

def create_cu3au_crystals():

    # 一つ目の結晶 Cu3Au のデータ
    lattice_params = {
        "lengths": [4.24596403, 4.24596403, 4.24596403],
        "angles": [90.00000000, 90.00000000, 90.00000000]
    }

    frac_coords_1 = [
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ]
    atom_types_1 = [29, 29, 29, 79]
    
    # 2つ目の結晶：Cu₃AuのCuの位置は同じ、Auの位置はランダム
    frac_coords_2 = [
        [0.5, 0.5, 0.5],  # Cu1
        [0.0, 0.0, 0.5],  # Cu2
        [0.5, 0.0, 0.0],  # Cu3
    ]

    # ランダムな位置にAuを配置
    # 各座標は0から1の範囲でランダムに選ばれます
    au_random_coords = [random.random() for _ in range(3)]

    # Auのランダムな座標を追加
    frac_coords_2.append(au_random_coords)

    atom_types_2 = [29, 29, 29, 79]
    
    # データをTorchテンソルに変換してバッチに格納
    loaded_batch = {
        'frac_coords': torch.tensor(frac_coords_1 + frac_coords_2),
        'atom_types': torch.tensor([atom_types_1, atom_types_2]),
        'lengths': torch.tensor([lattice_params["lengths"],lattice_params["lengths"]]),
        'angles': torch.tensor([lattice_params["angles"], lattice_params["angles"]]),
        'num_atoms': torch.tensor([len(frac_coords_1), len(frac_coords_2)]),
    }

    return loaded_batch

def visualize_structure_with_matplotlib(structure):
    """
    Visualize a crystal structure using matplotlib.
    
    Parameters:
    structure : object
        A structure object containing atom species and their fractional coordinates.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Group coordinates by atom species
    atom_coords = defaultdict(list)
    for site in structure:
        atom_coords[site.species_string].append(site.frac_coords)
    
    # Use colormap for different atom types
    colors = plt.cm.get_cmap('tab20', len(atom_coords))
    
    for idx, (atom_type, coords) in enumerate(atom_coords.items()):
        coords = np.array(coords)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                   s=50, color=colors(idx), label=atom_type, alpha=0.6)
    
    # Set plot limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X Fractional Coordinate')
    ax.set_ylabel('Y Fractional Coordinate')
    ax.set_zlabel('Z Fractional Coordinate')
    ax.set_title('Crystal Structure')

    # Show legend
    ax.legend(loc='upper right', fontsize=8)
    
    plt.show()

def scattering_factor_with_table(atom_number, K, af0_table):
    h, k, l = K
    filtered_row = af0_table[
        (af0_table['atom_num'] == atom_number.item()) &
        (af0_table['h'] == h) &
        (af0_table['k'] == k) &
        (af0_table['l'] == l)
    ]
    
    if not filtered_row.empty:
        f_q = filtered_row.iloc[0]['af0']
        return f_q
    else:
        print("エラー: 散乱因子が見つかりません")
        return 0  # エラー時のデフォルト値

def complex_sum_squared_with_scattering_factors_with_table(k, A, atom_types, af0_table, sum=[0.01, 0.01, 0.01]):
    i = complex(0, 1)
    pi = np.pi

    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    n = A.shape[0]
    result = 0.0
    for j in range(n):
        for m in range(n):
            f_j = scattering_factor_with_table(atom_types[j], k, af0_table)
            f_m = scattering_factor_with_table(atom_types[m], k, af0_table)
            diff = A[j] - A[m]
            k_sq = k ** 2
            r = np.dot(diff, k)
            r_c = np.dot(sum, k_sq)
            result += f_j * f_m * np.exp(2 * pi * i * r - 2 * (pi ** 2) * r_c)

    real_result = np.real(result)
    return real_result

def visualize_complex_sum(A, num_atoms, atom_types, af0_table):
    """
    Visualize the result of complex sum calculations over k1 and k2 using matplotlib.
    
    Parameters:
    A : ndarray
        The array representing atomic positions or other relevant structure.
    num_atoms : int
        Number of atoms in the structure.
    atom_types : list
        List of atom types in the structure.
    af0_table : ndarray
        Table of scattering factors or other relevant values.
    """
    # Define the k1 and k2 ranges
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    Z = np.zeros((len(k1_values), len(k2_values)))

    # Compute the complex sum for each k1, k2 pair
    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            k = np.array([-2, k1, k2])
            Z[i, j] = complex_sum_squared_with_scattering_factors_with_table(k, A, atom_types, af0_table)
    
    # Create the 3D plot
    X, Y = np.meshgrid(k1_values, k2_values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z.T, cmap='viridis', edgecolor='none')

    # Set plot labels and title
    ax.set_xlabel('k1', fontsize=10)
    ax.set_ylabel('k2', fontsize=10)
    ax.set_zlabel('|complex_sum|', fontsize=10)
    ax.set_title(f'Complex Sum for {atom_types}', fontsize=12)

    # Add a color bar to indicate the Z values
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Display the plot
    plt.show()



def main():
    # データの呼び出し、データをCPUにマッピング
    loaded_batch = create_cu3au_crystals()
    # 読み込んだデータを使用
    print(loaded_batch)

    num_crystals = loaded_batch['num_atoms'].size(0)  # バッチサイズ
    af0_table = get_fq()  # ループの外で一度だけ取得

    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
        start_index = sum(loaded_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        end_index = start_index + loaded_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        # i番目の結晶のデータを抽出
        first_frac_coords = loaded_batch['frac_coords'][start_index:end_index]
        first_atom_types = loaded_batch['atom_types'][i]
        first_lengths = loaded_batch['lengths'][i]
        first_angles = loaded_batch['angles'][i]
        num_atoms = loaded_batch['num_atoms'][i]

        # Latticeオブジェクトを生成（格子パラメータから）
        lattice = Lattice.from_parameters(first_lengths[0], first_lengths[1], first_lengths[2],
                                          first_angles[0], first_angles[1], first_angles[2])

        # pymatgenのStructureオブジェクトを作成
        structure = Structure(lattice, first_atom_types, first_frac_coords)
        print(structure)

        # 結晶構造を可視化
        #visualize_structure(structure)  # 可視化関数を呼び出し
        visualize_structure_with_matplotlib(structure)

        # Fを計算
        visualize_complex_sum(first_frac_coords, num_atoms, first_atom_types, af0_table)


if __name__ == "__main__":
    main()