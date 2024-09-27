from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import torch
from get_fq import get_fq
from collections import defaultdict

def visualize_structure_with_matplotlib(ax, structure, if_gt, comment, show_legend=True):
    """
    結晶構造を指定されたAxesにmatplotlibで可視化する関数
    """
    atom_coords = defaultdict(list)
    for site in structure:
        atom_coords[site.species_string].append(site.frac_coords)
    
    colors = plt.cm.get_cmap('tab20', len(atom_coords))
    for idx, (atom_type, coords) in enumerate(atom_coords.items()):
        coords = np.array(coords)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                   s=50, color=colors(idx), label=atom_type, alpha=0.6)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    ax.set_xlabel('X Fractional Coordinate', fontsize=10)
    ax.set_ylabel('Y Fractional Coordinate', fontsize=10)
    ax.set_zlabel('Z Fractional Coordinate', fontsize=10)
    ax.set_title(f'Crystal Structure: {if_gt}\n{comment}', fontsize=12)
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.legend_.remove() if ax.get_legend() else None

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

def complex_sum_squared_with_scattering_factors_with_table(k, A, atom_types, af0_table, sum=[0.02, 0.02, 0.02]):
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

    real_result = np.real(result) / 100
    return real_result

def visualize_complex_sum(ax, A, num_atoms, atom_types, af0_table):
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    Z = np.zeros((len(k1_values), len(k2_values)))

    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            k = np.array([0, k1, k2])
            Z[i, j] = complex_sum_squared_with_scattering_factors_with_table(k, A, atom_types, af0_table)
    
    X, Y = np.meshgrid(k1_values, k2_values)
    surf = ax.plot_surface(X, Y, Z.T, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('k1', fontsize=10)
    ax.set_ylabel('k2', fontsize=10)
    ax.set_zlabel('|complex_sum|', fontsize=10)
    ax.set_title(f'Complex Sum: {atom_types}', fontsize=12)
    fig = ax.get_figure()
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

def calc_loss(A_1, A_2, atom_types_1, atom_types_2, af0_table):
    print(A_1)
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    k3_values = np.arange(-2, 3, 1)
    Z_1 = np.zeros((len(k1_values), len(k2_values), len(k3_values)))
    Z_2 = np.zeros((len(k1_values), len(k2_values), len(k3_values)))

    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            for l, k3 in enumerate(k2_values):
                k = np.array([k1, k2, k3])
                Z_1[i, j, l] = complex_sum_squared_with_scattering_factors_with_table(k, A_1, atom_types_1, af0_table)
                Z_2[i, j, l] = complex_sum_squared_with_scattering_factors_with_table(k, A_2, atom_types_2, af0_table)

    # RMSEの計算
    rmse = np.sqrt(np.mean((Z_1 - Z_2) ** 2))
    print(f"RMSE: {rmse}")
    return rmse

def visualize_4_grid(i, comment, num_atoms, af0_table, tensor_1, tensor_2, true_structure, predicted_structure, true_frac_coords, true_atom_types, predicted_frac_coords, predicted_atom_types):
    calc_loss(true_frac_coords, predicted_frac_coords, true_atom_types, predicted_atom_types, af0_table)
    plt.plot(tensor_1[:, i], label=f'狙って生成: Crystal {i+1}', linestyle='-', alpha=0.7)
    plt.plot(tensor_2[:, i], label=f'ランダム生成: Crystal {i+1}', linestyle='--', alpha=0.7)
    # ラベルとタイトルを設定
    plt.xlabel('Time Step')
    plt.ylabel('Loss Value')
    #plt.title('損失が増えてしまった')
    plt.title('損失')

    # レジェンドを表示
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # プロットを表示
    plt.tight_layout()
    plt.show()
    
    # プロット用のFigureとAxesを作成（2行2列）
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), subplot_kw={'projection': '3d'}, constrained_layout=True)

    # 真の構造の可視化
    visualize_structure_with_matplotlib(axes[0, 0], true_structure, "GT", comment, show_legend=False)
    axes[0, 0].set_title('True Structure (GT)', fontsize=10)

    # 予測された構造の可視化
    visualize_structure_with_matplotlib(axes[0, 1], predicted_structure, "Generated", comment, show_legend=False)
    axes[0, 1].set_title('Predicted Structure (Generated)', fontsize=10)

    # 複素数和の可視化（真の構造）
    visualize_complex_sum(axes[1, 0], true_frac_coords, num_atoms, true_atom_types, af0_table)
    axes[1, 0].set_title('Complex Sum (GT)', fontsize=10)

    # 複素数和の可視化（予測された構造）
    visualize_complex_sum(axes[1, 1], predicted_frac_coords, num_atoms, predicted_atom_types, af0_table)
    axes[1, 1].set_title(f'Complex Sum (Generated)', fontsize=10)

    # 共通の凡例を追加
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc='upper right', fontsize=10)

    plt.show()  # 各結晶ごとに表示


def main():
    # ファイルパス
    file_1 = 'sample/sample_d1_48_100/loss_tensor.pt' # 狙って生成
    file_2 = 'sample/sample_d1_48_100_0/loss_tensor.pt' #ランダム生成

    # テンソルを読み込む
    tensor_1 = torch.load(file_1).numpy()
    tensor_2 = torch.load(file_2).numpy()

    print(tensor_1.shape)  # Expecting (100, 256)
    print(tensor_2.shape)  # Expecting (100, 256)

    # ロスが減ったもの
    #filtered_indices = (tensor_1[-1, :] < tensor_2[-1, :]).nonzero()[0]
    # ロスが増えたもの
    filtered_indices = (tensor_1[-1, :] > tensor_2[-1, :]).nonzero()[0]
    print(len(filtered_indices))

    # 真の結晶構造のデータの呼び出し、データをCPUにマッピング
    true_batch = torch.load('sample/sample_d1_48_100/batch.pt', map_location=torch.device('cpu'))
    # 予測された結晶構造のデータの呼び出し、データをCPUにマッピング
    predicted_batch = torch.load('sample/sample_d1_48_100/traj.pt', map_location=torch.device('cpu'))

    num_crystals = true_batch['num_atoms'].size(0)  # バッチサイズ

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    score = 0

    af0_table = get_fq()  # ループの外で一度だけ取得

    # バッチ内の全ての結晶に対してループ
    #for i in range(num_crystals):
    for i in filtered_indices:
        print(i)
        # 真の結晶構造のデータを抽出
        true_start_index = sum(true_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        true_end_index = true_start_index + true_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        num_atoms = true_batch['num_atoms'][i]

        true_frac_coords = true_batch['frac_coords'][true_start_index:true_end_index]
        true_atom_types = true_batch['atom_types'][true_start_index:true_end_index]
        true_lengths = true_batch['lengths'][i]
        true_angles = true_batch['angles'][i]

        true_lattice = Lattice.from_parameters(true_lengths[0], true_lengths[1], true_lengths[2],
                                               true_angles[0], true_angles[1], true_angles[2])
        true_structure = Structure(true_lattice, true_atom_types, true_frac_coords)

        # 予測された結晶構造のデータを抽出
        predicted_start_index = sum(predicted_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        predicted_end_index = predicted_start_index + predicted_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        predicted_frac_coords = predicted_batch['frac_coords'][predicted_start_index:predicted_end_index]
        predicted_atom_types = predicted_batch['atom_types'][predicted_start_index:predicted_end_index]
        predicted_lattice = predicted_batch['lattices'][i]

        predicted_structure = Structure(predicted_lattice, predicted_atom_types, predicted_frac_coords)

        # 構造の一致度を確認
        if matcher.fit(true_structure, predicted_structure):
            comment = "一致している"
            score += 1  # 一致した場合のスコアをインクリメント
            
        else:
            comment = "一致していない"
            visualize_4_grid(i, comment, num_atoms, af0_table, tensor_1, tensor_2, true_structure, predicted_structure, true_frac_coords, true_atom_types, predicted_frac_coords, predicted_atom_types)
            

    #print(f"一致率: {score / num_crystals:.2f}")
    print(f"一致率: {score / len(filtered_indices):.2f}")


if __name__ == "__main__":
    main()