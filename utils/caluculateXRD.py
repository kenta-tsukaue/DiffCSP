import torch
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

loaded_batch = torch.load('batch.pt', map_location=torch.device('cpu'))

num_crystals = loaded_batch['num_atoms'].size(0)  # バッチサイズ

for i in range(num_crystals):
    # ステップ1: 結晶構造の生成
    start_index = sum(loaded_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
    end_index = start_index + loaded_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

    # i番目の結晶のデータを抽出
    first_frac_coords = loaded_batch['frac_coords'][start_index:end_index]
    first_atom_types = loaded_batch['atom_types'][start_index:end_index]
    first_lengths = loaded_batch['lengths'][i]
    first_angles = loaded_batch['angles'][i]

    # Latticeオブジェクトを生成（格子パラメータから）
    lattice = Lattice.from_parameters(
        first_lengths[0], first_lengths[1], first_lengths[2],
        first_angles[0], first_angles[1], first_angles[2]
    )

    # pymatgenのStructureオブジェクトを作成
    structure = Structure(lattice, first_atom_types, first_frac_coords)

    # ステップ2: XRD計算器の設定
    xrd_calculator = XRDCalculator(wavelength="CuKa1")  # 銅Kα1放射線を使用

    # ステップ3: 回折パターンの計算
    pattern = xrd_calculator.get_pattern(structure, two_theta_range=(10, 90))

    # ステップ4: 結果の可視化
    plt.figure(figsize=(10, 6))
    plt.plot(pattern.x, pattern.y, label='XRD Pattern')
    plt.title('X-ray Diffraction Pattern')
    plt.xlabel('2θ [degree]')
    plt.ylabel('Intensity [arbitrary units]')
    plt.grid(True)
    plt.legend()
    plt.show()

