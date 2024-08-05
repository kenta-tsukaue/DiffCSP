import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

def plot_complete_unit_cell(ax, lattice, origin):
    """Plot all 12 edges of the unit cell from the given origin."""
    # Generate the ends of each lattice vector
    ends = [origin + v for v in lattice.matrix]
    # List to hold the start and end of each edge
    edges = [
        (origin, ends[0]), (origin, ends[1]), (origin, ends[2]),  # Origin to each vector end
        (ends[0], ends[0] + lattice.matrix[1]), (ends[0], ends[0] + lattice.matrix[2]),  # Vector 1 to others
        (ends[1], ends[1] + lattice.matrix[0]), (ends[1], ends[1] + lattice.matrix[2]),  # Vector 2 to others
        (ends[2], ends[2] + lattice.matrix[0]), (ends[2], ends[2] + lattice.matrix[1]),  # Vector 3 to others
        (ends[0] + lattice.matrix[1], ends[0] + lattice.matrix[1] + lattice.matrix[2]),  # Across the face
        (ends[1] + lattice.matrix[0], ends[1] + lattice.matrix[0] + lattice.matrix[2]),
        (ends[2] + lattice.matrix[0], ends[2] + lattice.matrix[0] + lattice.matrix[1])
    ]
    
    # Plot each edge in black
    for start, end in edges:
        ax.plot(*zip(start, end), 'k-', linewidth=2)

def main():
    # Load the batch data
    loaded_batch = torch.load('sample/d2_sample_gradual/traj.pt', map_location=torch.device('cpu'))
    #print(loaded_batch)
    # Extract data for the first crystal
    num_crystals = loaded_batch['num_atoms'].size(0)  # バッチサイズ
    

    for i in range(num_crystals):
        start_index = sum(loaded_batch['num_atoms'][:i])  # i番目の結晶の開始インデックス
        end_index = start_index + loaded_batch['num_atoms'][i]  # i番目の結晶の終了インデックス

        # i番目の結晶のデータを抽出
        first_frac_coords = loaded_batch['frac_coords'][start_index:end_index]
        first_atom_types = loaded_batch['atom_types'][start_index:end_index]
        lattice_tensor = loaded_batch['lattices'][i]
        lattice = Lattice(lattice_tensor.numpy())

        # pymatgenのStructureオブジェクトを作成
        structure = Structure(lattice, first_atom_types, first_frac_coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax._axis3don = False  # Hide the 3D axes

        # Generate a 3x3 grid of the lattice
        offsets = [np.dot([x, y, z], lattice.matrix) for x in range(-1, 2) for y in range(-1, 2) for z in [0]]
        for offset in offsets:
            plot_complete_unit_cell(ax, lattice, offset)
            
            # Plot the atoms
            for site in structure:
                pos = np.dot(site.frac_coords, lattice.matrix) + offset
                ax.scatter(*pos, color='gray', s=50)  # Red dots for atoms

        # Setting the title with dynamic atom count per unit cell
        atom_count = len(first_frac_coords)
        ax.set_title(f'3x3 Extended Crystal Structure of C{atom_count}')

        ax.set_xlabel('X [Å]')
        ax.set_ylabel('Y [Å]')
        ax.set_zlabel('Z [Å]')

        plt.show()

if __name__ == "__main__":
    main()
