import pandas as pd
import matplotlib.pyplot as plt

# エクセルファイルのパス
file_path = 'train.csv'

# エクセルファイルを読み込む
df = pd.read_csv(file_path)

# 'spacegroup.number' カラムのデータを集計する
count_data = df['spacegroup.number'].value_counts()

# 集計結果を表示
print(count_data)

# 集計結果をバー・チャートで可視化
plt.figure(figsize=(10, 8))
count_data.plot(kind='bar')
plt.title('Frequency of Each Spacegroup Number')
plt.xlabel('Spacegroup Number')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
