from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import torch


def main():
    # 真の結晶構造のデータの呼び出し、データをCPUにマッピング
    true_batch = torch.load('sample/d2_sample_gradual/batch.pt', map_location=torch.device('cpu'))
    # 予測された結晶構造のデータの呼び出し、データをCPUにマッピング
    predicted_batch = torch.load('sample/d2_sample_gradual/traj.pt', map_location=torch.device('cpu'))

    num_crystals = true_batch['num_atoms'].size(0)  # バッチサイズ

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
        # 真の結晶構造のデータを抽出
        true_start_index = sum(true_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        true_end_index = true_start_index + true_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

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
            print(f"Structure {i+1} matches.")
        else:
            print(f"Structure {i+1} does not match.")

if __name__ == "__main__":
    main()