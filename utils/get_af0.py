import CFML_api
from CFML_api.API_Reflections_Utilities import ReflectionList
import numpy as np
import pandas as pd

# 原子記号と原子番号の対応を辞書に保存
atom_symbol_to_number = {
    "H ": 1, "He": 2, "Li": 3, "Be": 4, "B ": 5, "C ": 6, "N ": 7, "O ": 8, "F ": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P ": 15, "S ": 16, "Cl": 17, "Ar": 18,
    "K ": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V ": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, 
    "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y ": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, 
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I ": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72,
    "Ta": 73, "W ": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81,
    "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U ": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
    "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107,
    "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115,
    "Lv": 116, "Ts": 117, "Og": 118
}

# 原子記号を与えて原子番号を返す関数
def get_atomic_number(symbol):
    return atom_symbol_to_number.get(symbol, "Invalid atom symbol")

def af0():
    cellv = np.asarray([4.24596403, 4.24596403, 4.24596403], dtype='float32')
    angl = np.asarray([90,90,90], dtype='float32')

    cell = CFML_api.Cell(cellv, angl)

    # Create list from string
    print("========\nCreate atom_list from string")
    dat = [
        'loop_                     ',
        '_atom_site_label          ',
        '_atom_site_fract_x        ',
        '_atom_site_fract_y        ',
        '_atom_site_fract_z        ',
        '_atom_site_U_iso_or_equiv ',
        'Co 0.00265771 0.00000000 0.00000000 0.00450',  # Coは重いので小さめの値
        'Tl 0.50015703 0.50000000 0.50000000 0.00380',  # Tlも重いので小さめの値
        'N1 0.50108143 0.00000000 0.50000000 0.01200',  # Nは軽いので大きめの値
        'N2 0.50108143 0.50000000 0.00000000 0.01200',  # Nの別の位置でも同様
        'O 0.00050506 0.50000000 0.50000000 0.01050'    # 他の軽い原子も少し大きめ
    ]
    atom_list = CFML_api.AtomList(dat)

    dat = [
    'Title SrTiO3',
    'Npatt 1',
    'Patt_1 XRAY_2THE  1.54056    1.54056    1.00      0.0        135.0',
    'UVWXY        0.025  -0.00020   0.01200   0.00150  0.00465',
    'STEP         0.05 ',
    'Backgd       50.000']

    job_info = CFML_api.JobInfo(dat)
    a=CFML_api.SpaceGroup(1)

    reflection_list = ReflectionList(cell=cell, spg=a, lfriedel=True, job=job_info)
    reflection_list.compute_structure_factors(a, atom_list, job_info)
    data_dict = reflection_list.compute_af0(a, atom_list, job_info)
    print(reflection_list.print_description())

    # 辞書データからaf0の1次元配列を取得
    af0_1d = data_dict['af0']  # 'data_dict' が辞書オブジェクト

    # 1次元目と2次元目のサイズ
    n_atoms = len(data_dict['atom'])
    n_reflections = len(data_dict['h'])

    # af0の2次元配列に復元
    af0_2d = np.array(af0_1d).reshape(n_atoms, n_reflections)

    #print(af0_2d[3])
    #print(data_dict['h'])

    # df作成
    data = []
    columns = ["atom", "atom_num", "h", "k", "l", "af0"]
    for idx, atom in enumerate(data_dict['atom']):
        for i in range(len(data_dict['h'])):
            kari_list = [atom, get_atomic_number(atom), data_dict['h'][i], data_dict['k'][i], data_dict['l'][i], af0_2d[idx][i]]
            data.append(kari_list)
    # DataFrameの作成
    df = pd.DataFrame(data, columns=columns)
    print(df["atom_num"])
    return df
