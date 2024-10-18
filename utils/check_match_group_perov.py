"""
=======================================[実行例]=======================================
python check_match_group_perov.py --dir=sample/sample_d1_48_104_1 --batch_id=0
=====================================================================================
"""
import argparse
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import torch
from collections import defaultdict
import warnings

def parse_argments():
    parser = argparse.ArgumentParser(
        description="一致度チェックの引数設定"
    )
    parser.add_argument(
        '--dir',
        type=str,
        help="ディレクトリ"
    )
    parser.add_argument(
        '--batch_id',
        type=int,
        help='バッチインデックス'
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_argments()
    # 警告メッセージを抑制（必要に応じてコメントアウト）
    warnings.filterwarnings("ignore", category=FutureWarning)

    # グループ定義: 組成式の最後の3つの原子番号に基づく
    group_definitions = {
        tuple([7, 7, 7]): 'Group1 [7,7,7]',
        tuple([8, 8, 8]): 'Group2 [8,8,8]',
        tuple([7, 7, 8]): 'Group3 [7,7,8]',
        tuple([7, 8, 8]): 'Group4 [7,8,8]',
        tuple([7, 8, 9]): 'Group5 [7,8,9]',
        tuple([8, 8, 9]): 'Group6 [8,8,9]',
        tuple([16, 8, 8]): 'Group7 [16,8,8]'
    }

    # カウント用辞書の初期化
    # 各グループごとに新手法のみ一致、両方一致、既存手法のみ一致のカウントを保持
    group_counts = {
        group_name: {'new_matched_only': 0, 'both_matched': 0, 'old_matched_only': 0, 'total': 0} 
        for group_name in group_definitions.values()
    }
    group_counts['Others'] = {'new_matched_only': 0, 'both_matched': 0, 'old_matched_only': 0, 'total': 0}  # 定義外のグループ用

    # 全体のカウント初期化
    overall_counts = {'new_matched_only': 0, 'both_matched': 0, 'old_matched_only': 0, 'total': 0}

    # データの読み込み
    true_batch = torch.load(f'{args.dir}/batch_{args.batch_id}.pt', map_location=torch.device('cpu'))
    predicted_new_batch = torch.load(f'{args.dir}/traj_new_{args.batch_id}.pt', map_location=torch.device('cpu'))
    predicted_old_batch = torch.load(f'{args.dir}/traj_{args.batch_id}.pt', map_location=torch.device('cpu'))

    num_crystals = true_batch['num_atoms'].size(0)  # バッチサイズ

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

    # バッチ内の全ての結晶に対してループ
    for i in range(num_crystals):
        # 真の結晶構造のデータを抽出
        true_start_index = sum(true_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        true_end_index = true_start_index + true_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        true_frac_coords = true_batch['frac_coords'][true_start_index:true_end_index].tolist()
        true_atom_types = true_batch['atom_types'][true_start_index:true_end_index].tolist()
        true_lengths = true_batch['lengths'][i].tolist()
        true_angles = true_batch['angles'][i].tolist()

        true_lattice = Lattice.from_parameters(
            true_lengths[0], true_lengths[1], true_lengths[2],
            true_angles[0], true_angles[1], true_angles[2]
        )
        true_structure = Structure(
            true_lattice, 
            [int(atom) for atom in true_atom_types], 
            true_frac_coords
        )

        # 新手法の予測データを抽出
        new_start_index = sum(predicted_new_batch['num_atoms'][:i])
        new_end_index = new_start_index + predicted_new_batch['num_atoms'][i]

        new_frac_coords = predicted_new_batch['frac_coords'][new_start_index:new_end_index].tolist()
        new_atom_types = predicted_new_batch['atom_types'][new_start_index:new_end_index].tolist()
        new_lattice = predicted_new_batch['lattices'][i].tolist()

        new_structure = Structure(
            Lattice(new_lattice),
            [int(atom) for atom in new_atom_types],
            new_frac_coords
        )

        # 既存手法の予測データを抽出
        old_start_index = sum(predicted_old_batch['num_atoms'][:i])
        old_end_index = old_start_index + predicted_old_batch['num_atoms'][i]

        old_frac_coords = predicted_old_batch['frac_coords'][old_start_index:old_end_index].tolist()
        old_atom_types = predicted_old_batch['atom_types'][old_start_index:old_end_index].tolist()
        old_lattice = predicted_old_batch['lattices'][i].tolist()

        old_structure = Structure(
            Lattice(old_lattice),
            [int(atom) for atom in old_atom_types],
            old_frac_coords
        )

        # グループの判定
        if len(true_atom_types) >= 3:
            last_three = tuple(true_atom_types[-3:])
        else:
            last_three = tuple(true_atom_types)  # 原子数が3未満の場合

        group_name = group_definitions.get(last_three, 'Others')

        # カウント更新
        group_counts[group_name]['total'] += 1
        overall_counts['total'] += 1

        # 構造の一致度を確認
        new_matched = matcher.fit(true_structure, new_structure)
        old_matched = matcher.fit(true_structure, old_structure)

        # カウントのロジックを修正
        if new_matched and old_matched:
            group_counts[group_name]['both_matched'] += 1
            overall_counts['both_matched'] += 1
        elif new_matched and not old_matched:
            group_counts[group_name]['new_matched_only'] += 1
            overall_counts['new_matched_only'] += 1
        elif not new_matched and old_matched:
            group_counts[group_name]['old_matched_only'] += 1
            overall_counts['old_matched_only'] += 1
        # 両方一致しない場合はカウントしない

    # 結果の表示
    print("各グループごとの一致率と一致カウント (新手法一致率 / 既存手法一致率 / どちらか一致率)")
    print("(新手法一致率 / 既存手法一致率 / どちらか一致率) および (新手法のみ / 両方 / 既存手法のみ)")
    for group, counts in group_counts.items():
        if counts['total'] > 0:
            new_match_rate = (counts['new_matched_only'] + counts['both_matched']) / counts['total']
            old_match_rate = (counts['old_matched_only'] + counts['both_matched']) / counts['total']
            either_match_rate = (counts['new_matched_only'] + counts['both_matched'] + counts['old_matched_only']) / counts['total']
            print(f"{group}:")
            print(f"  一致率: 新手法一致率: {new_match_rate:.2%} / 既存手法一致率: {old_match_rate:.2%} / どちらか一致率: {either_match_rate:.2%}")
            print(f"  カウント: 新手法のみ: {counts['new_matched_only']} / 両方: {counts['both_matched']} / 既存手法のみ: {counts['old_matched_only']}")
        else:
            print(f"{group}: データなし")

    # 全体の結果
    print("\n全体の一致率と一致カウント (新手法一致率 / 既存手法一致率 / どちらか一致率)")
    print("(新手法一致率 / 既存手法一致率 / どちらか一致率) および (新手法のみ / 両方 / 既存手法のみ)")
    if overall_counts['total'] > 0:
        overall_new_match_rate = (overall_counts['new_matched_only'] + overall_counts['both_matched']) / overall_counts['total']
        overall_old_match_rate = (overall_counts['old_matched_only'] + overall_counts['both_matched']) / overall_counts['total']
        overall_either_match_rate = (overall_counts['new_matched_only'] + overall_counts['both_matched'] + overall_counts['old_matched_only']) / overall_counts['total']
        print(f"全体:")
        print(f"  一致率: 新手法一致率: {overall_new_match_rate:.2%} / 既存手法一致率: {overall_old_match_rate:.2%} / どちらか一致率: {overall_either_match_rate:.2%}")
        print(f"  カウント: 新手法のみ: {overall_counts['new_matched_only']} / 両方: {overall_counts['both_matched']} / 既存手法のみ: {overall_counts['old_matched_only']}")
    else:
        print("全体: データなし")

if __name__ == "__main__":
    main()