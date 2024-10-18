from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import torch
from get_fq import get_fq
from collections import defaultdict

def find_difference(match_list_new, match_list_random):
    # Find elements in match_list_random that are not in match_list_new
    difference_list = [item for item in match_list_random if item not in match_list_new]
    return difference_list

def find_matching_elements(filtered_indices, akka_list):
    # Find elements in akka_list that are also in filtered_indices
    matching_elements = [item for item in akka_list if item in filtered_indices]
    return matching_elements

def main():
    # 真の結晶構造のデータの呼び出し、データをCPUにマッピング
    true_batch = torch.load('sample/sample_d1_48_104_1/batch_0.pt', map_location=torch.device('cpu'))
    # 予測された結晶構造のデータの呼び出し、データをCPUにマッピング
    predicted_batch = torch.load('sample/sample_d1_48_104_1/traj_new_0.pt', map_location=torch.device('cpu'))

    # 予測された結晶構造のデータの呼び出し、データをCPUにマッピング
    predicted_batch_random = torch.load('sample/sample_d1_48_104_1/traj_0.pt', map_location=torch.device('cpu'))

    num_crystals = true_batch['num_atoms'].size(0)  # バッチサイズ

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    score_random = 0
    score_new = 0

    match_list_random = []
    match_list_new = []

    # バッチ内の全ての結晶に対してループ
    #for i in range(num_crystals):
    for i in range(num_crystals):
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

        # ランダム生成された結晶構造のデータを抽出
        predicted_frac_coords_random = predicted_batch_random['frac_coords'][predicted_start_index:predicted_end_index]
        predicted_atom_types = predicted_batch_random['atom_types'][predicted_start_index:predicted_end_index]
        predicted_lattice_random = predicted_batch_random['lattices'][i]

        predicted_structure_random = Structure(predicted_lattice_random, predicted_atom_types, predicted_frac_coords_random)

        # 構造の一致度を確認
        if matcher.fit(true_structure, predicted_structure):
            score_new += 1  # 一致した場合のスコアをインクリメント
            match_list_new.append(i)
        # 構造の一致度を確認
        if matcher.fit(true_structure, predicted_structure_random):
            score_random += 1  # 一致した場合のスコアをインクリメント
            match_list_random.append(i)
            

    #print(f"一致率: {score / num_crystals:.2f}")
    print(f"一致率(ランダム): {score_random / num_crystals:.2f}")
    print(f"一致率(新手法): {score_new / num_crystals:.2f}")
    print(score_new, score_random)

    # ファイルパス
    file_1 = 'sample/sample_d1_48_104_2/loss_tensor_new_0.pt' # 狙って生成
    file_2 = 'sample/sample_d1_48_104_2/loss_tensor_0.pt' #ランダム生成

    # テンソルを読み込む
    tensor_1 = torch.load(file_1).numpy()
    tensor_2 = torch.load(file_2).numpy()

    print(tensor_1.shape)  # Expecting (100, 256)
    print(tensor_2.shape)  # Expecting (100, 256)

    # ロスが減ったもの
    filtered_indices = (tensor_1[-1, :] < tensor_2[-1, :]).nonzero()[0]
    
    ryouka_list = find_difference(match_list_random, match_list_new) # ランダム生成で一致していなくて、新手法で一致するもの
    akka_list = find_difference(match_list_new, match_list_random) # 新手法で一致しなくて、ランダム生成で一致するもの
    print("ランダム生成で一致する数:", len(match_list_random))
    print("新手法で一致する数:", len(match_list_new))
    print("良化数", len(ryouka_list))
    print("悪化数",len(akka_list))
    #print(akka_list)

    loss_low_list = find_matching_elements(filtered_indices, akka_list)
    print(len(loss_low_list))


if __name__ == "__main__":
    main()