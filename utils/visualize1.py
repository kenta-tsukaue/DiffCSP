import os
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.vis.structure_vtk import StructureVis  # VTKベースの可視化

def visualize_structure(structure):
    """結晶構造を可視化するヘルパー関数"""
    vis = StructureVis()
    vis.set_structure(structure)
    vis.show()

def main():
    # データの呼び出し、データをCPUにマッピング
    loaded_batch = torch.load('batch.pt', map_location=torch.device('cpu'))
    # 読み込んだデータを使用
    print(loaded_batch)

    num_crystals = loaded_batch['num_atoms'].size(0)  # バッチサイズ

    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
        start_index = sum(loaded_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        end_index = start_index + loaded_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        # i番目の結晶のデータを抽出
        first_frac_coords = loaded_batch['frac_coords'][start_index:end_index]
        first_atom_types = loaded_batch['atom_types'][start_index:end_index]
        first_lengths = loaded_batch['lengths'][i]
        first_angles = loaded_batch['angles'][i]

        # Latticeオブジェクトを生成（格子パラメータから）
        lattice = Lattice.from_parameters(first_lengths[0], first_lengths[1], first_lengths[2],
                                          first_angles[0], first_angles[1], first_angles[2])

        # pymatgenのStructureオブジェクトを作成
        structure = Structure(lattice, first_atom_types, first_frac_coords)

        # 結晶構造を可視化
        visualize_structure(structure)  # 可視化関数を呼び出し

if __name__ == "__main__":
    main()
