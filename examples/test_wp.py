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

@wp.kernel
def copy_tiled_2(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):
    i,j = wp.tid()
    t = wp.tile_load(a, shape = (2,3))
    u = wp.tile(wp.vec3f(1.0, 2.0, 3.0))
    u = wp.tile_transpose(u)
    u = wp.tile_broadcast(u, shape=(2, 3))
    # you are allowed to just add tiles like that
    u = u + t
    wp.tile_store(b, u)




N = 2
a = torch.rand(N, 3)
b = wp.zeros((N, 3), dtype=float)
# print("a", a)
# wp.launch(copy_reg, dim=[N,3], inputs=[a], outputs=[b])
# wp.launch_tiled(copy_tiled_2, dim=[N,3], inputs=[a, b], block_dim=3)
# print("b", b)


@wp.kernel
def copy_tiled_3(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float)):
    i,j = wp.tid()
    t = wp.tile_load(a, shape = (3,3))
    u = wp.tile(wp.mat33(1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0,
                        7.0, 8.0, 9.0))
    u = wp.tile_squeeze(u)
    # u = u + t
    wp.tile_store(b, u)

N = 3
a = torch.rand(N, 3)
b = wp.zeros((N, 3), dtype=float)
print("a", a)
wp.launch_tiled(copy_tiled_3, dim=[N,3], inputs=[a, b], block_dim=1)
print("b", b)

