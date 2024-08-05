import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import torch


def complex_sum(k, A):
    """
    3次元ベクトル k と (n x 3) の行列 A を受け取り、
    行列 A の各行とベクトル k の内積 r を計算し、
    exp(2 * pi * i * r) を足し合わせて、その和を返す関数。

    Parameters:
    k (np.ndarray): 3次元ベクトル
    A (np.ndarray): (n x 3) の行列

    Returns:
    complex: 複素数の和
    """
    i = complex(0, 1)
    pi = np.pi

    # 各行とベクトル k の内積を計算
    r = np.dot(A, k)

    # exp(2 * pi * i * r) の和を計算
    result = np.sum(np.exp(2 * pi * i * r))

    return result

def visualize_complex_sum(A, num_atoms):
    # kの範囲設定
    k1_values = np.arange(-2, 3, 1)
    k2_values = np.arange(-2, 3, 1)
    # 結果を格納する配列
    Z = np.zeros((len(k1_values), len(k2_values)))
    print(Z)

    # k1とk2を動かしてcomplex_sumの値を計算
    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            k = np.array([k1, k2, 0])
            Z[i, j] = np.abs(complex_sum(k, A))

    print(Z)

    # プロット
    X, Y = np.meshgrid(k1_values, k2_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z.T, cmap='viridis')

    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    ax.set_zlabel(f'|complex_sum|')
    ax.set_title(f'3D Plot of complex_sum /C{num_atoms}')
    plt.show()


#  debugのため見易い結晶構造を作る。
#  lattice vectorはunit vector。cellは-1/2<=x,y,z<=+1/2とする。
#  8点の座標を10種類決める。
r1 = np.sqrt(0.5)
square1 = [[np.cos(theta)*r1, np.sin(theta)*r1] for theta in np.arange(np.pi/4, 2*np.pi, np.pi/2)]
square2 = [[np.cos(theta)*r1, np.sin(theta)*r1] for theta in np.arange(0, 2*np.pi, np.pi/2)]
cube1 = [[x, y, z] for z in [-0.5, 0.5] for x, y in square1]
cube2 = [[x, y, -0.5] for x, y in square1] + [[x, y, 0.5] for x, y in square2]

triangle1 = [[np.cos(theta)*r1, np.sin(theta)*r1] for theta in np.arange(0, 2*np.pi, np.pi*2/3)]
triangle1.append([0, 0])
triangle2 = [[np.cos(theta)*r1, np.sin(theta)*r1] for theta in np.arange(np.pi/3, 2*np.pi, np.pi*2/3)]
triangle2.append([0, 0])
prism1 = [[x, y, z] for z in [-0.5, 0.5] for x, y in triangle1]
prism2 = [[x, y, -0.5] for x, y in triangle1] + [[x, y, 0.5] for x, y in triangle2]

hexagon1 = [[np.cos(theta)*r1, np.sin(theta)*r1, 0] for theta in np.arange(0, 2*np.pi, np.pi*1/3)]
hexagon1.append([0, 0, 0.5])
hexagon1.append([0, 0, -0.5])

hexagon2 = [[0, np.cos(theta)*r1, np.sin(theta)*r1] for theta in np.arange(0, 2*np.pi, np.pi*1/3)]
hexagon2.append([0.5, 0, 0])
hexagon2.append([-0.5, 0, 0])

octagon1 = [[np.cos(theta)*r1, np.sin(theta)*r1, 0] for theta in np.arange(0, 2*np.pi, np.pi*1/4)]
octagon2 = [[0, np.cos(theta)*r1, np.sin(theta)*r1] for theta in np.arange(0, 2*np.pi, np.pi*1/4)]

zigzag = list(zip(np.arange(-3/8, 0.5, 1/4), [-1/8, 1/8]*2))
zigzag1 = [[x, y, z] for z in [-0.25, 0.25] for x, y in zigzag]
zigzag2 = [[x, y, -0.25] for x, y in zigzag] + [[x, -y, 0.25] for x, y in zigzag]


scale_factor = 0.6
offset = 0.5 * (1 - scale_factor)

r1 = np.sqrt(0.5) * scale_factor

# 正八角形 (octagon)
octagon = [[(np.cos(theta) * r1 + 1) / 2, (np.sin(theta) * r1 + 1) / 2, 0.5] for theta in np.arange(0, 2 * np.pi, np.pi / 4)]
octagon = [[x * scale_factor + offset, y * scale_factor + offset, z] for x, y, z in octagon]

# 直線 (line)
line = [[0.5, 0.5, (i + 4) * 0.125] for i in range(-4, 4)]
line = [[x * scale_factor + offset, y * scale_factor + offset, z] for x, y, z in line]

# 正六面体 (cube)
square1 = [[(np.cos(theta) * r1 + 1) / 2, (np.sin(theta) * r1 + 1) / 2] for theta in np.arange(np.pi / 4, 2 * np.pi, np.pi / 2)]
cube = [[x, y, (z + 1) / 2] for z in [-0.5, 0.5] for x, y in square1]
cube = [[x * scale_factor + offset, y * scale_factor + offset, z * scale_factor + offset] for x, y, z in cube]

# ジグザグ構造 (zigzag)
zigzag = [[(x + 1) / 2, (y + 1) / 2, (z + 0.25) * 2] for z in [-0.25, 0.25] for x, y in zip(np.arange(-3 / 8, 0.5, 1 / 4), [-1 / 8, 1 / 8] * 2)]
zigzag = [[x * scale_factor + offset, y * scale_factor + offset, z * scale_factor + offset] for x, y, z in zigzag]

# 位置にGaussian noiseを加えて水増しする。
std = 0.01
for cell0 in [octagon[:8], line[:8], cube[:8], zigzag[:8]]:
    print('start new cell')
    for _ in range(1):
        cell1 = []
        for xyz0 in cell0:
            xyz1 = np.array(xyz0) + np.random.randn(3) * std
            xyz1 = xyz1 % 1.0  # 座標を0~1の範囲にシフトして、周期境界条件を適用
            cell1.append(xyz1)
    res = cell1  # 最後の結果を描画
    print(res)
    # Convert the numpy arrays to a 8x3 torch tensor
    first_frac_coords = torch.tensor(res)

    # Create an 8-dimensional torch tensor with all elements equal to 6
    first_atom_types = torch.full((8,), 6)

    # Create a tensor with 8 as its value
    num_atoms = torch.tensor(8)

    # Create a 3x3 torch tensor representing the lattice
    lattice = torch.tensor([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    # Print the tensors
    print("first_frac_coords:\n", first_frac_coords)
    print("first_atom_types:\n", first_atom_types)
    print("num_atoms:\n", num_atoms)
    print("lattice:\n", lattice)

    # Fを計算
    #visualize_complex_sum(first_frac_coords, num_atoms)

    xs = [x for x, _, _ in res]
    ys = [y for _, y, _ in res]
    zs = [z for _, _, z in res]
    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')])
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=10, range=[0, 1]),
        yaxis=dict(nticks=10, range=[0, 1]),
        zaxis=dict(nticks=10, range=[0, 1]),
        aspectmode='cube'  # 各軸のスケールを同じに設定
    ))
    fig.show()