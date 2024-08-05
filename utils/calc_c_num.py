import pandas as pd

# CSVファイルの読み込み
with open('train.csv', 'r') as file:
    data = file.read()

# データセクションごとに分割
data_sections = data.split("data_C")[1:]

carbon_counts = []

# 各データセクションを解析
for section in data_sections:
    lines = section.strip().split('\n')
    
    carbon_count = 0
    for line in lines:
        if line.startswith('_chemical_formula_sum'):
            parts = line.split()
            for part in parts:
                if part.startswith('C') and part[1:].isdigit():
                    carbon_count = int(part[1:])
                    break
            if carbon_count > 0:
                break
    carbon_counts.append(carbon_count)

# 炭素原子の数ごとのデータの数を集計
carbon_count_distribution = {}
for count in carbon_counts:
    if count in carbon_count_distribution:
        carbon_count_distribution[count] += 1
    else:
        carbon_count_distribution[count] = 1

# 炭素原子の数ごとのデータの数をデータ数の降順にソート
sorted_carbon_count_distribution = sorted(carbon_count_distribution.items(), key=lambda x: x[1], reverse=True)

# 結果の表示
print("炭素原子の数ごとのデータの数 (データ数の降順):")
for count, num in sorted_carbon_count_distribution:
    print(f"炭素原子の数 {count}: {num} 個のデータ")