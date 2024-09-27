import os
import numpy as np
import torch
import pandas as pd

scattering_factor_path = "scattering_factors.xlsx"

def scattering_factor_torch_batch_from_file(atom_types):
    """
    PyTorchを使用して散乱因子を計算する関数(バッチ対応）。

    Parameters:
    - atom_types (torch.Tensor): 形状 (C, A_max) のテンソル。各原子の原子番号。
    - scattering_factor_path (str): 散乱因子データのExcelファイルのパス。

    Returns:
    - f_q (torch.Tensor): 形状 (C, A_max, 125) のテンソル。各原子の散乱因子。
    """
    # デバイスの取得（atom_typesと同じデバイスに配置）
    device = atom_types.device

    # Excelファイルから散乱因子データを読み込む
    scattering_factor_df = pd.read_excel(scattering_factor_path)

    # 必要な列のみを抽出
    scattering_factor_df = scattering_factor_df[['atom_num', 'af0']]

    # atom_numごとにaf0をリストとして集約
    grouped = scattering_factor_df.groupby('atom_num')['af0'].apply(list)

    # atom_numの最大値を取得
    max_atom_num = grouped.index.max()

    # af0の長さを確認（すべて125であることを前提）
    af0_length = grouped.iloc[0].__len__()  # 最初のatom_numのaf0の長さを取得
    assert all(len(af0_list) == af0_length for af0_list in grouped), "すべてのatom_numでaf0の長さが一致していません。"

    # 最大原子番号に基づいてルックアップテーブルを初期化（0も含めるためmax_atom_num + 1）
    lookup_table = torch.zeros((max_atom_num + 1, af0_length), dtype=torch.float32, device=device)

    # 各atom_numに対してaf0をルックアップテーブルに格納
    for atom_num, af0_list in grouped.items():
        if atom_num == 0:
            continue  # atom_numが0の場合は既に0で初期化されているためスキップ
        # atom_numが整数であることを確認
        if not isinstance(atom_num, int):
            raise ValueError(f"atom_numが整数ではありません: {atom_num}")
        # af0_listをテンソルに変換し、ルックアップテーブルに代入
        lookup_table[atom_num] = torch.tensor(af0_list, dtype=torch.float32, device=device)

    # atom_typesが0の場合はルックアップテーブルの0番目（全て0）を使用
    # その他の場合は対応するaf0を取得
    # atom_typesの形状は (C, A_max) で、出力f_qの形状は (C, A_max, 125)
    f_q = lookup_table[atom_types]  # 高度なインデックス付けを使用

    return f_q

def get_fq(batch):
    """
    バッチ内の全ての結晶に対して構造因子のx_t微分を計算する関数。
    
    Args:
        batch (dict): バッチ情報を含む辞書。必要なキーは 'num_atoms' と 'atom_types'。
        m (torch.Tensor): m 座標テンソル [Total_atoms, 3]
        c (torch.Tensor): c 座標テンソル [Total_atoms, 3]
        delm_delx_t (torch.Tensor): delm/delx_t テンソル [Total_atoms]
    
    Returns:
        np.ndarray: 計算された Z テンソル [C, 5, 5, 5, num_atoms_max, 3]
    """
    device = batch['frac_coords'].device if 'frac_coords' in batch else 'cpu'
    num_atoms = batch['num_atoms'].to('cpu')  # [C]
    num_crystals = num_atoms.size(0)
    num_atoms_max = torch.max(num_atoms).item()
    atom_types = batch['atom_types'].to(device)  # [Total_atoms]
    
    padded_atom_types = torch.zeros((num_crystals, num_atoms_max), dtype=torch.long, device=device)


    start = 0
    for i in range(num_crystals):
        n = num_atoms[i].item()
        padded_atom_types[i, :n] = atom_types[start:start+n]
        start += n
    
    # 散乱因子の計算
    # f_i: [C, A_max, K]
    fq = scattering_factor_torch_batch_from_file(padded_atom_types)  # [C, A_max, K]
    #(f_i.shape)
    return fq


def get_fq():
    # Excelファイルから散乱因子データを読み込む
    scattering_factor_df = pd.read_excel(scattering_factor_path)

    return scattering_factor_df