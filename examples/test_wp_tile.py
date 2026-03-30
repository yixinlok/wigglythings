import warp as wp
import numpy as np
TILE_SIZE = wp.constant(17)
TILE_THREADS = 64

@wp.kernel
def compute(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):

    # obtain our block index
    i = wp.tid()

    # load a row from global memory
    t = wp.tile_load(a, (TILE_SIZE, 1), offset=(i*TILE_SIZE, 0))
    s = wp.tile_ones((3, TILE_SIZE), dtype=float)

    # cooperatively compute the sum of the tile elements; s is a single element tile
    out = wp.tile_zeros((3,1), dtype=float)
    wp.tile_matmul(s, t, out)

    # store s in global memory
    wp.tile_store(b, out, offset=(0, i))

N = 7

a_np = np.arange(N).reshape(-1, 1) * np.ones((1, TILE_SIZE), dtype=float)
a = wp.array(a_np, dtype=float, device="cuda:0")
b = wp.zeros((3, N), dtype=float, device="cuda:0")

# dimension is 256. blocks i runs from 1, 2, 3, .... 256
# block dim is 32. so within each thread, we split it into 32 threads to solve the matmul of 1x256 256x1  
# 
wp.launch_tiled(compute, dim=[a.shape[0]], inputs=[a, b], block_dim=256, device="cuda:0")
# wp.launch(compute, dim=[a.shape[0]], inputs=[a, b], device="cuda:0")
# print(f"a = {a}")
print(f"b = {b}")