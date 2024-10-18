import torch
import matplotlib.pyplot as plt
import japanize_matplotlib

# ファイルパス
file_1 = 'sample/sample_d1_48_104_1/batch_loss_tensor_0.pt'
file_2 = 'sample/sample_d1_48_104_1/batch_loss_tensor_new_0.pt'

# テンソルを読み込む
tensor_1 = torch.load(file_1)
tensor_2 = torch.load(file_2)

# テンソルをnumpy配列に変換
tensor_1 = tensor_1.numpy()
tensor_2 = tensor_2.numpy()

# プロット
plt.figure(figsize=(10, 5))
plt.plot(tensor_1, label='ランダム生成')
plt.plot(tensor_2, label='狙って生成')

# ラベルとタイトルを設定
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of Two Loss Tensors')
plt.legend()

# プロットを表示
plt.show()