import os
import argparse
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import torch

def load_batch(batch_path):
    """真のデータをロードします。"""
    return torch.load(batch_path, map_location=torch.device('cpu'))

def load_traj(traj_path):
    """予測されたデータをロードします。"""
    return torch.load(traj_path, map_location=torch.device('cpu'))

def get_iteration_files(directory, prefix):
    """指定されたディレクトリ内の指定されたプレフィックスのファイルを取得し、イテレーション順にソートします。"""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.pt')]
    # イテレーション番号でソート
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[-1].split('.pt')[0]))
    return files_sorted

def main():
    parser = argparse.ArgumentParser(description="結晶構造の一致度を計算します。")
    parser.add_argument('--directory', type=str, help='データが格納されているディレクトリのパス')
    args = parser.parse_args()
    directory = args.directory

    # 真のデータのファイルを取得
    batch_files = get_iteration_files(directory, 'batch_')
    if not batch_files:
        print(f"真のデータファイルが見つかりません。ディレクトリ: {directory}")
        return

    # 予測データのファイルを取得
    traj_files = get_iteration_files(directory, 'traj_')
    traj_new_files = get_iteration_files(directory, 'traj_new_')

    # イテレーション番号の抽出
    def extract_iteration(file_list, prefix):
        iterations = []
        for f in file_list:
            try:
                iter_num = int(f.replace(prefix, '').replace('.pt', ''))
                iterations.append(iter_num)
            except ValueError:
                continue
        return sorted(iterations)

    batch_iterations = extract_iteration(batch_files, 'batch_')
    traj_iterations = extract_iteration(traj_files, 'traj_')
    traj_new_iterations = extract_iteration(traj_new_files, 'traj_new_')

    # 真のデータと予測データのイテレーション番号を一致させる
    common_iterations = set(batch_iterations) & set(traj_iterations) & set(traj_new_iterations)
    if not common_iterations:
        print("一致するイテレーション番号が見つかりません。")
        return

    # ソートされたイテレーション番号
    common_iterations = sorted(common_iterations)

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

    # ランダム生成と新手法の一致度をカウントする変数
    results = {
        'random': {'matches': 0, 'total': 0, 'per_num_atoms': {}},
        'new_method': {'matches': 0, 'total': 0, 'per_num_atoms': {}}
    }

    for idx, iter_num in enumerate(common_iterations, 1):
        print(f'進捗 : {idx} / {len(common_iterations)}')
        batch_filename = f"batch_{iter_num}.pt"
        traj_filename = f"traj_{iter_num}.pt"
        traj_new_filename = f"traj_new_{iter_num}.pt"

        batch_path = os.path.join(directory, batch_filename)
        traj_path = os.path.join(directory, traj_filename)
        traj_new_path = os.path.join(directory, traj_new_filename)

        # 真のデータのロード
        true_batch = load_batch(batch_path)

        # ランダム生成データのロード
        if os.path.exists(traj_path):
            predicted_random_batch = load_traj(traj_path)
        else:
            print(f"ランダム生成データファイルが見つかりません: {traj_path}")
            continue

        # 新手法データのロード
        if os.path.exists(traj_new_path):
            predicted_new_batch = load_traj(traj_new_path)
        else:
            print(f"新手法データファイルが見つかりません: {traj_new_path}")
            continue

        num_crystals = true_batch['num_atoms'].size(0)  # バッチサイズ

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
            true_structure = Structure(true_lattice, true_atom_types, true_frac_coords)

            # ランダム生成された結晶構造のデータを抽出
            predicted_random_start = sum(predicted_random_batch['num_atoms'][:i])
            predicted_random_end = predicted_random_start + predicted_random_batch['num_atoms'][i]

            predicted_random_frac = predicted_random_batch['frac_coords'][predicted_random_start:predicted_random_end].tolist()
            predicted_random_atoms = predicted_random_batch['atom_types'][predicted_random_start:predicted_random_end].tolist()
            predicted_random_lattice = predicted_random_batch['lattices'][i].tolist()

            predicted_random_structure = Structure(
                predicted_random_lattice,
                predicted_random_atoms,
                predicted_random_frac
            )

            # 新手法によって生成された結晶構造のデータを抽出
            predicted_new_start = sum(predicted_new_batch['num_atoms'][:i])
            predicted_new_end = predicted_new_start + predicted_new_batch['num_atoms'][i]

            predicted_new_frac = predicted_new_batch['frac_coords'][predicted_new_start:predicted_new_end].tolist()
            predicted_new_atoms = predicted_new_batch['atom_types'][predicted_new_start:predicted_new_end].tolist()
            predicted_new_lattice = predicted_new_batch['lattices'][i].tolist()

            predicted_new_structure = Structure(
                predicted_new_lattice,
                predicted_new_atoms,
                predicted_new_frac
            )

            # 原子数を取得
            num_atoms = true_batch['num_atoms'][i].item()

            # ランダム生成の一致度を確認
            if num_atoms not in results['random']['per_num_atoms']:
                results['random']['per_num_atoms'][num_atoms] = {'matches': 0, 'total': 0}
            results['random']['total'] += 1
            results['random']['per_num_atoms'][num_atoms]['total'] += 1
            if matcher.fit(true_structure, predicted_random_structure):
                results['random']['matches'] += 1
                results['random']['per_num_atoms'][num_atoms]['matches'] += 1

            # 新手法の一致度を確認
            if num_atoms not in results['new_method']['per_num_atoms']:
                results['new_method']['per_num_atoms'][num_atoms] = {'matches': 0, 'total': 0}
            results['new_method']['total'] += 1
            results['new_method']['per_num_atoms'][num_atoms]['total'] += 1
            if matcher.fit(true_structure, predicted_new_structure):
                results['new_method']['matches'] += 1
                results['new_method']['per_num_atoms'][num_atoms]['matches'] += 1

    # 結果の表示
    for method in ['random', 'new_method']:
        matches = results[method]['matches']
        total = results[method]['total']
        if total > 0:
            rate = matches / total
            method_name = 'ランダム生成' if method == 'random' else '新手法'
            print(f"{method_name} 一致率: {rate:.4f} ({matches}/{total})")
        else:
            method_name = 'ランダム生成' if method == 'random' else '新手法'
            print(f"{method_name} に該当するデータがありません。")

        # 原子数ごとの一致率を表示
        print(f"{method_name} の原子数ごとの一致率:")
        per_num_atoms = results[method]['per_num_atoms']
        for num_atoms in sorted(per_num_atoms.keys()):
            atom_matches = per_num_atoms[num_atoms]['matches']
            atom_total = per_num_atoms[num_atoms]['total']
            atom_rate = atom_matches / atom_total if atom_total > 0 else 0
            print(f"  原子数 {num_atoms}: 一致率 {atom_rate:.4f} ({atom_matches}/{atom_total})")

if __name__ == "__main__":
    main()