import os
import argparse
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure, Lattice
import torch

def create_cu3au_crystals():
    # 一つ目の結晶：32個の全ての原子がランダム配置
    atom_types_1 = [29] * 6 + [79] * 2  # Cu (24個), Au (8個)
    atom_types_2 = [29, 79, 29, 29, 29, 29, 29, 79]
    # 2つ目の結晶：最初の24個のCuはランダム、最後の8個のAuは規則的に配置
    
    # Auは規則的に配置 (8つのユニットセルの中心にAuを配置, 0.25単位)
    au_regular_coords = [
        [0.25, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.75, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0.25, 0.5, 0.5],
        [0.75, 0.5, 0.5],
    ]
    # 格子パラメータは両方の結晶で同じ
    lattice_params = {
        "lengths": [4.24596403 * 2, 4.24596403, 4.24596403],
        "angles": [90.00000000, 90.00000000, 90.00000000]
    }

    # データをTorchテンソルに変換してバッチに格納
    loaded_batch = {
        'frac_coords': torch.tensor(au_regular_coords + au_regular_coords),
        'atom_types': torch.tensor([atom_types_1, atom_types_2]),
        'lengths': torch.tensor([lattice_params["lengths"], lattice_params["lengths"]]),
        'angles': torch.tensor([lattice_params["angles"], lattice_params["angles"]]),
        'num_atoms': torch.tensor([len(au_regular_coords), len(au_regular_coords)]),
    }

    return loaded_batch

def load_batch(batch_path):
    """真のデータをロードします。"""
    return torch.load(batch_path, map_location=torch.device('cpu'))

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


    # ソートされたイテレーション番号
    common_iterations = sorted(batch_iterations)

    matcher = StructureMatcher(ltol=0.3, stol=0.05, angle_tol=10)

    # ランダム生成と新手法の一致度をカウントする変数
    results = {
        'random': {'matches': 0, 'total': 0},
        'new_method': {'matches': 0, 'total': 0}
    }

    for iter_num in common_iterations:
        print(f'進捗 : {iter_num} / {len(common_iterations)}')
        batch_filename = f"batch_{iter_num}.pt"

        batch_path = os.path.join(directory, batch_filename)

        # 真のデータのロード
        true_batch = load_batch(batch_path)
        cho_batch = create_cu3au_crystals()
        

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

            # i番目の結晶のデータを抽出
            first_frac_coords = cho_batch['frac_coords'][0:8]
            first_atom_types = cho_batch['atom_types'][0]
            first_lengths = cho_batch['lengths'][0]
            first_angles = cho_batch['angles'][0]
            num_atoms = cho_batch['num_atoms'][0]

            # Latticeオブジェクトを生成（格子パラメータから）
            lattice = Lattice.from_parameters(first_lengths[0], first_lengths[1], first_lengths[2],
                                            first_angles[0], first_angles[1], first_angles[2])
            structure = Structure(lattice, first_atom_types, first_frac_coords)


            # 超格子構造との一致度を確認
            if matcher.fit(true_structure, structure):
                results['random']['matches'] += 1
            results['random']['total'] += 1

    # 結果の表示
    for method in ['random']:
        matches = results[method]['matches']
        total = results[method]['total']
        if total > 0:
            rate = matches / total
            method_name = 'ランダム生成' if method == 'random' else '新手法'
            print(f"{method_name} 一致率: {rate:.4f} ({matches}/{total})")
        else:
            method_name = 'ランダム生成' if method == 'random' else '新手法'
            print(f"{method_name} に該当するデータがありません。")

if __name__ == "__main__":
    main()