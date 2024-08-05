import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# 訓練データの結晶構造を生成
def generate_training_data(std=0.05):
    r1 = np.sqrt(0.5)
    square1 = [[np.cos(theta) * r1, np.sin(theta) * r1] for theta in np.arange(np.pi/4, 2*np.pi, np.pi/2)]
    square2 = [[np.cos(theta) * r1, np.sin(theta) * r1] for theta in np.arange(0, 2*np.pi, np.pi/2)]
    cube1 = [[x, y, z] for z in [-0.5, 0.5] for x, y in square1]
    cube2 = [[x, y, -0.5] for x, y in square1] + [[x, y, 0.5] for x, y in square2]

    triangle1 = [[np.cos(theta) * r1, np.sin(theta) * r1] for theta in np.arange(0, 2*np.pi, np.pi*2/3)]
    triangle1.append([0, 0])
    triangle2 = [[np.cos(theta) * r1, np.sin(theta) * r1] for theta in np.arange(np.pi/3, 2*np.pi, np.pi*2/3)]
    triangle2.append([0, 0])
    prism1 = [[x, y, z] for z in [-0.5, 0.5] for x, y in triangle1]
    prism2 = [[x, y, -0.5] for x, y in triangle1] + [[x, y, 0.5] for x, y in triangle2]

    hexagon1 = [[np.cos(theta) * r1, np.sin(theta) * r1, 0] for theta in np.arange(0, 2*np.pi, np.pi*1/3)]
    hexagon1.append([0, 0, 0.5])
    hexagon1.append([0, 0, -0.5])

    hexagon2 = [[0, np.cos(theta) * r1, np.sin(theta) * r1] for theta in np.arange(0, 2*np.pi, np.pi*1/3)]
    hexagon2.append([0.5, 0, 0])
    hexagon2.append([-0.5, 0, 0])

    octagon1 = [[np.cos(theta) * r1, np.sin(theta) * r1, 0] for theta in np.arange(0, 2*np.pi, np.pi*1/4)]
    octagon2 = [[0, np.cos(theta) * r1, np.sin(theta) * r1] for theta in np.arange(0, 2*np.pi, np.pi*1/4)]

    zigzag = list(zip(np.arange(-3/8, 0.5, 1/4), [-1/8, 1/8] * 2))
    zigzag1 = [[x, y, z] for z in [-0.25, 0.25] for x, y in zigzag]
    zigzag2 = [[x, y, -0.25] for x, y in zigzag] + [[x, -y, 0.25] for x, y in zigzag]

    training_data = [cube1, cube2, prism1, prism2, hexagon1, hexagon2, octagon1, octagon2, zigzag1, zigzag2]

    structures = []
    for cell0 in training_data:
        cell1 = []
        for xyz0 in cell0:
            xyz1 = np.array(xyz0) + np.random.randn(3) * std
            xyz1 = (xyz1 + 0.5) % 1.0  # 座標を0~1の範囲にシフトして、周期境界条件を適用
            cell1.append(xyz1)
        frac_coords = np.array(cell1)
        atom_types = [6] * len(frac_coords)
        lattice = np.eye(3)
        structures.append(Structure(lattice, atom_types, frac_coords))
    return structures

# 訓練データを生成
training_structures = generate_training_data()

# 生成された結晶構造を読み込み、訓練データと比較する
def main():
    # データの呼び出し、データをCPUにマッピング
    loaded_batch = torch.load('sample/d1_sample_sample10/traj.pt', map_location=torch.device('cpu'))

    num_crystals = loaded_batch['num_atoms'].size(0)  # バッチサイズ
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    score_list = []


    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
        total_score = 0
        start_index = sum(loaded_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        end_index = start_index + loaded_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        # i番目の結晶のデータを抽出
        first_frac_coords = loaded_batch['frac_coords'][start_index:end_index].numpy()
        first_atom_types = loaded_batch['atom_types'][start_index:end_index].numpy()
        lattice = loaded_batch['lattices'][i].numpy()

        # pymatgenのStructureオブジェクトを作成
        generated_structure = Structure(lattice, first_atom_types, first_frac_coords)

        # 訓練データと比較
        for training_structure in training_structures:
            if matcher.fit(generated_structure, training_structure):
                total_score += 1
        
        score_list.append(total_score)

    print("Total Scores:", score_list)

if __name__ == "__main__":
    main()