import warp as wp
import numpy as np

num_instances = 3
num_instances_v = 4
num_modes = 2

num_instances = wp.constant(num_instances)
num_instances_v = wp.constant(num_instances_v)
num_instances_v3 = wp.constant(num_instances_v*3)

num_modes = wp.constant(num_modes)


# @wp.kernel()
# def displace_tiled(
#         eigenvectors: wp.array2d(dtype=float),
#         q_cur: wp.array2d(dtype=float),
#         displace: wp.array2d(dtype=float),
#         displace_t: wp.array3d(dtype=float)):
#     tid = wp.tid()
#     f = wp.tile(eigenvectors)
#     evs = wp.tile_load(eigenvectors, shape=(num_instances_v3, num_modes))
#     qs = wp.tile_load(q_cur, shape=(num_modes, 1))
#     d = wp.tile_zeros(shape=(num_instances_v3, 1), dtype=wp.float32)
    
#     wp.tile_matmul(evs, qs, d)
#     d = wp.tile_squeeze(d)
#     wp.tile_store(displace[tid], d)


@wp.kernel()
def loss(eigenvectors: wp.array(dtype=wp.mat((num_instances_v*3,num_modes), dtype=float)),
        q_cur: wp.array(dtype=wp.vec(length=num_modes, dtype=float)),
        displace: wp.array(dtype=wp.vec(length=num_instances_v*3, dtype=float)),
        displace_t: wp.array(dtype=wp.mat((num_instances_v,3), dtype=float))):
    tid = wp.tid()
    displace[tid] = eigenvectors[0]@q_cur[tid]
    displace_t[tid] = wp.types.matrix(displace[tid], shape=(num_instances_v,3))

    # for i in range(num_instances_v*3):
    #     displace_t[tid][i//num_instances_v][i%3] = displace[tid][i]
    
ev = np.zeros((num_instances, num_instances_v*3, num_modes), dtype=np.float32)
ev[0] = np.random.rand(num_instances_v*3, num_modes).astype(np.float32)
ev = wp.from_numpy(ev, device="cuda:0")


q_cur = wp.from_numpy(np.random.rand(num_instances, num_modes).astype(np.float32), device="cuda:0")
displace = wp.from_numpy(np.zeros((num_instances, num_instances_v*3), dtype=np.float32), device="cuda:0")
displace_t = wp.from_numpy(np.zeros((num_instances, num_instances_v, 3), dtype=np.float32), device="cuda:0")

wp.launch(loss, dim=num_instances, inputs=[ev, q_cur], outputs=[displace, displace_t], device="cuda:0")

