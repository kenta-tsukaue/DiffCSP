import os
import argparse
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def load_batch(batch_path):
    """真のデータをロードします。"""
    return torch.load(batch_path, map_location=torch.device('cpu'))

def load_traj(traj_path):
    """予測されたデータをロードします。"""
    return torch.load(traj_path, map_location=torch.device('cpu'))

def get_iteration_files(directory, prefix):
    """指定されたディレクトリ内の指定されたプレフィックスのファイルを取得し、イテレーション順にソートします。"""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.pt')]
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[-1].split('.pt')[0]))
    return files_sorted

def match_structures(matcher, true_structure, predicted_structure, timeout=30):
    """StructureMatcherの一致度をタイムアウト付きで確認します。"""
    with ThreadPoolExecutor() as executor:
        future = executor.submit(matcher.fit, true_structure, predicted_structure)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print("一致度測定がタイムアウトしました")
            return False

def main():
    parser = argparse.ArgumentParser(description="結晶構造の一致度を計算します。")
    parser.add_argument('--directory', type=str, help='データが格納されているディレクトリのパス', required=True)
    args = parser.parse_args()
    directory = args.directory

    batch_files = get_iteration_files(directory, 'batch_')
    if not batch_files:
        print(f"真のデータファイルが見つかりません。ディレクトリ: {directory}")
        return

    traj_files = get_iteration_files(directory, 'traj_')
    traj_new_files = get_iteration_files(directory, 'traj_new_')

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

    common_iterations = set(batch_iterations) & set(traj_iterations) & set(traj_new_iterations)
    if not common_iterations:
        print("一致するイテレーション番号が見つかりません。")
        return

    common_iterations = sorted(common_iterations)
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)

    results = {'random': 0, 'new_method': 0}
    total_iterations = len(common_iterations)

    for idx, iter_num in enumerate(common_iterations, 1):
        print(f'進捗 : {idx} / {total_iterations} ')
        if idx == 804 or idx == 157 or idx == 1752 or idx == 2559:
            continue
        batch_filename = f"batch_{iter_num}.pt"
        traj_filename = f"traj_{iter_num}.pt"
        traj_new_filename = f"traj_new_{iter_num}.pt"

        batch_path = os.path.join(directory, batch_filename)
        traj_path = os.path.join(directory, traj_filename)
        traj_new_path = os.path.join(directory, traj_new_filename)

        true_batch = load_batch(batch_path)

        if os.path.exists(traj_path):
            predicted_random_batch = load_traj(traj_path)
        else:
            print(f"ランダム生成データファイルが見つかりません: {traj_path}")
            continue

        if os.path.exists(traj_new_path):
            predicted_new_batch = load_traj(traj_new_path)
        else:
            print(f"新手法データファイルが見つかりません: {traj_new_path}")
            continue

        num_crystals = true_batch['num_atoms'].size(0)

        random_matched = False
        new_method_matched = False

        for i in range(num_crystals):
            true_start_index = sum(true_batch['num_atoms'][:i])
            true_end_index = true_start_index + true_batch['num_atoms'][i]

            true_frac_coords = true_batch['frac_coords'][true_start_index:true_end_index].tolist()
            true_atom_types = true_batch['atom_types'][true_start_index:true_end_index].tolist()
            true_lengths = true_batch['lengths'][i].tolist()
            true_angles = true_batch['angles'][i].tolist()

            true_lattice = Lattice.from_parameters(
                true_lengths[0], true_lengths[1], true_lengths[2],
                true_angles[0], true_angles[1], true_angles[2]
            )
            true_structure = Structure(true_lattice, true_atom_types, true_frac_coords)

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

            if not random_matched and match_structures(matcher, true_structure, predicted_random_structure, timeout=30):
                random_matched = True
                print("ランダム生成データのマッチが見つかりました")

            if not new_method_matched and match_structures(matcher, true_structure, predicted_new_structure, timeout=30):
                new_method_matched = True
                print("新手法のマッチが見つかりました")

            if random_matched and new_method_matched:
                break

        if random_matched:
            results['random'] += 1
        if new_method_matched:
            results['new_method'] += 1

    for method in ['random', 'new_method']:
        matches = results[method]
        rate = matches / total_iterations if total_iterations > 0 else 0
        method_name = 'ランダム生成' if method == 'random' else '新手法'
        print(f"{method_name} 一致率: {rate:.4f} ({matches}/{total_iterations})")

if __name__ == "__main__":
    main()