import numpy as np
import warp as wp

TILE_SIZE = wp.constant(255)
TILE_THREADS = 64

A = wp.constant(2)
B = wp.constant(3)
C = wp.constant(6)
@wp.kernel
def compute(a: wp.array3d(dtype=float), b: wp.array2d(dtype=float)):

    # obtain our block index
    i = wp.tid()

    # load a row from global memory
    t = wp.tile_load(a[i], (A,B))
    # t = wp.tile_transpose(t)
    # t = wp.tile_reshape(t, (C, 1))
    t = wp.tile_reshape(t, (1, C))
    t = wp.tile_transpose(t)

    # store the result back to global memory
    wp.tile_store(b, t, offset=(0,i))

device = "cpu"
N = 2
a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)  # shape (N,A,B)
print("a shape:", a.shape)
a = wp.array(a, device=device)  # copy to GPU
b = wp.zeros((C,N), dtype=wp.float32, device=device)  # shape (C,N)

wp.launch_tiled(compute, dim=[a.shape[0]], inputs=[a, b], block_dim=2, device=device)

print(f"b = {b}")