import os
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.vis.structure_vtk import StructureVis  # VTKベースの可視化
import plotly.graph_objects as go


def visualize_structure(structure):
    """結晶構造を可視化するヘルパー関数"""
    vis = StructureVis()
    vis.set_structure(structure)
    vis.show()

def visualize_structure_plotly(structure):
    """結晶構造を可視化するヘルパー関数"""
    frac_coords = structure.frac_coords
    xs = frac_coords[:, 0]
    ys = frac_coords[:, 1]
    zs = frac_coords[:, 2]

    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')])
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=10, range=[0, 1]),
        yaxis=dict(nticks=10, range=[0, 1]),
        zaxis=dict(nticks=10, range=[0, 1]),
        aspectmode='cube'  # 各軸のスケールを同じに設定
    ))
    fig.show()

def main():
    # データの呼び出し、データをCPUにマッピング
    loaded_batch = torch.load('sample/d2_sample_gradual/traj.pt', map_location=torch.device('cpu'))
    # 読み込んだデータを使用
    #print(loaded_batch)

    num_crystals = loaded_batch['num_atoms'].size(0)  # バッチサイズ

    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
        start_index = sum(loaded_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        end_index = start_index + loaded_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        # i番目の結晶のデータを抽出
        first_frac_coords = loaded_batch['frac_coords'][start_index:end_index]
        print(first_frac_coords)
        first_atom_types = loaded_batch['atom_types'][start_index:end_index]
        lattice = loaded_batch['lattices'][i]
        print(lattice)

        # pymatgenのStructureオブジェクトを作成
        structure = Structure(lattice, first_atom_types, first_frac_coords)

        # 結晶構造を可視化
        visualize_structure(structure)  # 可視化関数を呼び出し

if __name__ == "__main__":
    main()