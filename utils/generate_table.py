import CFML_api
from CFML_api.API_Reflections_Utilities import ReflectionList
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm  # 進捗表示用
import re

# 原子記号と原子番号の対応を辞書に保存（スペースを保持）
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
def get_atomic_number(label):
    # ラベルから原子記号を抽出（数字を除去）
    match = re.match(r"([A-Za-z]+)", label)
    if match:
        symbol = match.group(1)
        # 辞書のキーに合うようにスペースを追加
        if symbol in ["H", "B", "C", "N", "O", "F", "P", "S", "K", "V", "Y", "I", "W", "U"]:
            symbol += " "  # スペースを追加
        return atom_symbol_to_number.get(symbol, "Invalid atom symbol")
    else:
        return "Invalid label"

def generate_scattering_factors(atom_symbols, output_filename="scattering_factors.xlsx"):
    """
    指定された原子リストの各原子について散乱因子を計算し、Excelに保存します。
    
    Parameters:
    - atom_symbols: list of str
        使用する原子の記号のリスト（例: ["Co", "Tl", "N", "O", "C", "Si", "Fe", "Mg", "Al", "Na"]）
    - output_filename: str
        保存するExcelファイルの名前
    """
    
    # 結果を保存するリスト
    all_data = []
    
    # セルパラメータの設定（固定値、必要に応じて変更）
    cellv = np.asarray([4.24596403, 4.24596403, 4.24596403], dtype='float32')
    angl = np.asarray([90, 90, 90], dtype='float32')
    cell = CFML_api.Cell(cellv, angl)
    
    # JobInfo の作成（固定値、必要に応じて変更）
    dat_job = [
        'Title GeneratedStructure',
        'Npatt 1',
        'Patt_1 XRAY_2THE  1.54056    1.54056    1.00      0.0        135.0',
        'UVWXY        0.025  -0.00020   0.01200   0.00150  0.00465',
        'STEP         0.05 ',
        'Backgd       50.000'
    ]
    job_info = CFML_api.JobInfo(dat_job)
    
    # SpaceGroup の作成（例として空間群1を使用）
    space_group = CFML_api.SpaceGroup(1)
    
    # 反射リストの設定（必要に応じて変更）
    # ここでは例として特定の反射指数を使用します。必要に応じて調整してください。
    # 例えば、h, k, l の範囲を指定
    hkl_range = range(-2, 3)  # -2から2まで
    reflections = [(h, k, l) for h in hkl_range for k in hkl_range for l in hkl_range if not (h == k == l == 0)]
    
    # 反射指数ごとに反射リストを作成
    reflection_indices = reflections  # リストとして保持
    
    # 進捗表示のためにtqdmを使用
    for atom in tqdm(atom_symbols, desc="Processing atoms"):
        # 各原子の位置を設定（単位セル内の任意の位置を使用）
        # ここでは単純に原点に配置
        fractional_positions = [
            [0.0, 0.0, 0.0]  # 各原子を単位セルの原点に配置
        ]
        
        # atom_site_label を生成
        atom_site_label = atom  # ラベルに数字を追加する必要はありません
        
        # dat リストの作成
        dat_atoms = [
            'loop_                     ',
            '_atom_site_label          ',
            '_atom_site_fract_x        ',
            '_atom_site_fract_y        ',
            '_atom_site_fract_z        ',
            f"{atom_site_label} {fractional_positions[0][0]:.8f} {fractional_positions[0][1]:.8f} {fractional_positions[0][2]:.8f}"
        ]
        print(dat_atoms)
        
        # AtomList の作成
        atom_list = CFML_api.AtomList(dat_atoms)
        
        # ReflectionList の作成と計算
        reflection_list = ReflectionList(cell=cell, spg=space_group, lfriedel=True, job=job_info)
        reflection_list.compute_structure_factors(space_group, atom_list, job_info)
        
        # Reflectionごとにaf0を計算
        data_dict = reflection_list.compute_af0(space_group, atom_list, job_info)
        
        # 辞書データからaf0の1次元配列を取得
        af0_1d = data_dict['af0']  # 'data_dict' が辞書オブジェクト
        
        # 1次元目と2次元目のサイズ
        n_atoms = len(data_dict['atom'])
        n_reflections = len(data_dict['h'])
        
        # af0の2次元配列に復元
        af0_2d = np.array(af0_1d).reshape(n_atoms, n_reflections)
        
        # df作成
        data = []
        columns = ["atom", "atom_num", "h", "k", "l", "af0"]
        for idx, atom_label in enumerate(data_dict['atom']):
            for i in range(n_reflections):
                kari_list = [
                    atom_label, 
                    get_atomic_number(atom_label), 
                    data_dict['h'][i], 
                    data_dict['k'][i], 
                    data_dict['l'][i], 
                    af0_2d[idx][i]
                ]
                data.append(kari_list)
        
        # DataFrameの作成
        df = pd.DataFrame(data, columns=columns)
        all_data.append(df)
    
    # 全てのデータを結合
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 重複を排除（必要に応じて）
    final_df = final_df.drop_duplicates()

    # h, k, l 列を数値として認識させる（必要に応じて）
    final_df['h'] = pd.to_numeric(final_df['h'], errors='coerce')
    final_df['k'] = pd.to_numeric(final_df['k'], errors='coerce')
    final_df['l'] = pd.to_numeric(final_df['l'], errors='coerce')

    # h, k, l の順に昇順で並べ替え
    df_sorted = final_df.sort_values(by=['atom_num', 'h', 'k', 'l'], ascending=[True, True, True, True])

    # Excelファイルに保存
    df_sorted.to_excel(output_filename, index=False)
    print(f"Data has been successfully saved to {output_filename}")

# 使用例
if __name__ == "__main__":
    # 使用する原子のリストを定義（全ての原子を含む）
    atom_list = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf"
    ]
    print(len(atom_list))
    # 関数を実行してExcelに保存
    generate_scattering_factors(atom_list, "scattering_factors.xlsx")


# Es ,"Fm", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"