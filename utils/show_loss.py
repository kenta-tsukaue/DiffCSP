import torch
import matplotlib.pyplot as plt

# ファイルパス
file_1 = 'sample/d1_43_17/loss_tensor.pt'
file_2 = 'sample/d1_43_18/loss_tensor.pt'

# テンソルを読み込む
tensor_1 = torch.load(file_1)
tensor_2 = torch.load(file_2)

# テンソルをnumpy配列に変換
tensor_1 = tensor_1.numpy()
tensor_2 = tensor_2.numpy()

# プロット
plt.figure(figsize=(10, 5))
plt.plot(tensor_1, label='d1_43_17')
plt.plot(tensor_2, label='d1_43_18')

# ラベルとタイトルを設定
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of Two Loss Tensors')
plt.legend()

# プロットを表示
plt.show()