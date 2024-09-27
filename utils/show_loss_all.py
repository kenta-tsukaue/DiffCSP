import torch
import matplotlib.pyplot as plt

# ファイルパス
file_1 = 'sample/sample_d1_48_100_0/loss_tensor.pt'

# テンソルを読み込む
tensor_1 = torch.load(file_1)
print(tensor_1.shape)  # Expected (100, 256)

# テンソルをnumpy配列に変換
tensor_1 = tensor_1.numpy()

# 各タイムステップの値を最初のタイムステップの値で割って正規化
normalized_tensor = tensor_1 / tensor_1[0, :]

# 最終タイムステップの値が最初のタイムステップの値より大きい結晶のみを選択
filtered_crystals = normalized_tensor[:, tensor_1[-1, :] > tensor_1[0, :]]

# プロット
plt.figure(figsize=(10, 5))

# フィルタリングされた結晶の正規化されたタイムステップでの値をプロット
for i in range(filtered_crystals.shape[1]):  # Plot only filtered crystals
    plt.plot(filtered_crystals[:, i], label=f'Crystal {i+1}', alpha=0.5)

# ラベルとタイトルを設定
plt.xlabel('Time Step')
plt.ylabel('Normalized Loss Value')
plt.title('Normalized Loss Values for Crystals with Final Value > Initial Value')

# プロットを表示
plt.show()