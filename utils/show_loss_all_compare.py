import torch
import matplotlib.pyplot as plt
import japanize_matplotlib


# ファイルパス
file_1 = 'sample/sample_d1_48_103_0/loss_tensor_new_0.pt' # 狙って生成
file_2 = 'sample/sample_d1_48_103_0/loss_tensor_0.pt' #ランダム生成

# テンソルを読み込む
tensor_1 = torch.load(file_1).numpy()
tensor_2 = torch.load(file_2).numpy()

print(tensor_1.shape)  # Expecting (100, 256)
print(tensor_2.shape)  # Expecting (100, 256)

# ロスが減ったもの
filtered_indices = (tensor_1[-1, :] < tensor_2[-1, :]).nonzero()[0]
# ロスが増えたもの
# filtered_indices = (tensor_1[-1, :] > tensor_2[-1, :]).nonzero()[0]
print(len(filtered_indices))

# 各結晶ごとに、tensor_1とtensor_2を同時に表示
plt.figure(figsize=(10, 5))

for idx in filtered_indices:
    plt.plot(tensor_1[:, idx], label=f'狙って生成: Crystal {idx+1}', linestyle='-', alpha=0.7)
    plt.plot(tensor_2[:, idx], label=f'ランダム生成: Crystal {idx+1}', linestyle='--', alpha=0.7)
    # ラベルとタイトルを設定
    plt.xlabel('Time Step')
    plt.ylabel('Loss Value')
    #plt.title('損失が増えてしまった')
    plt.title('損失')

    # レジェンドを表示
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # プロットを表示
    plt.tight_layout()
    plt.show()