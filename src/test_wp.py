import warp as wp
import torch 
import numpy as np

wp.init()
TILE_SIZE = wp.constant(1)
TILE_THREADS = 64

@wp.kernel
def copy_tiled(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):
    i,j = wp.tid()
    t = wp.tile_load(a[i], shape = TILE_SIZE, offset=j)
    wp.tile_store(b[i], t, offset=j)

@wp.kernel
def copy_reg(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):
    i,j = wp.tid()
    b[i][j] = a[i][j]


N = 2

a = torch.rand(N, 3)
b = wp.zeros((N, 3), dtype=float)
print("a shape", a)
wp.launch(copy_reg, dim=[N,3], inputs=[a], outputs=[b])
# wp.launch_tiled(copy_tiled, dim=[N,3], inputs=[a, b], block_dim=3)
print("b", b)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = a.to(device)
print("a device", a)
