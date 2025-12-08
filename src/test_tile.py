import numpy as np

num_v = 5
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # shape (3,)
pinned = [2]

# fill with x
forcing = np.tile(x, (num_v, 1))               # shape (num_v, 3)

# mask pinned -> set to zero
mask = np.zeros(num_v, dtype=bool)
mask[pinned] = True
forcing[mask] = np.array([4.0, 5.0, 6.0], dtype=np.float32)

# flattened version if phi_inv expects a 1D vector:
forcing_flat = forcing.ravel().T 

print("forcing:\n", forcing_flat)