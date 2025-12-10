import warp as wp
import numpy as np

num_instances = 10
num_instances_v = 3
num_modes = 2

# for i in range(ix.num_instances):
#     displace = bi.eigenvectors @ cpu_q_cur[i]
#     displace = np.reshape(displace, (3, -1)).T
#     displaces.append(displace)

# [a, b, c, d, e ,f]

# [a, b], [c, d], [e, f]

# [a, c, e], [b, d, f]

@wp.kernel()
def reshape3(displace_in: wp.array(dtype=wp.vec(length=num_instances_v*3, dtype=float)),
            # displace_out: wp.array(dtype=wp.vec3(dtype=float)),
            out: wp.array(dtype=wp.float32)):
    tid = wp.tid()

    out[tid] = displace_in[tid][0]
    # for i in range(num_instances_v):
    #     displace_out[tid* num_instances_v + i] = wp.vec3(displace_in[tid][i*3 + 0],
    #                                                       displace_in[tid][i*3 + 1],
    #                                                       displace_in[tid][i*3 + 2])
    
@wp.kernel()
def loss(eigenvectors: wp.array(dtype=wp.mat((num_instances_v*3,num_modes), dtype=float)),
        q_cur: wp.array(dtype=wp.vec(length=num_modes, dtype=float)),
        displace: wp.array(dtype=wp.vec(length=num_instances_v*3, dtype=float))):
    tid = wp.tid()
    displace[tid] = eigenvectors[0]@q_cur[tid]
    
ev = np.zeros((num_instances, num_instances_v*3, num_modes), dtype=np.float32)
ev[0] = np.random.rand(num_instances_v*3, num_modes).astype(np.float32)
ev = wp.from_numpy(ev, device="cuda:0")
print("ev shape:", ev.shape)
q_cur = wp.from_numpy(np.random.rand(num_instances, num_modes).astype(np.float32), device="cuda:0")
displace = wp.from_numpy(np.zeros((num_instances), dtype=np.float32), device="cuda:0")

wp.launch(loss, dim=num_instances, inputs=[ev, q_cur], outputs=[displace], device="cuda:0")
print("displace:", displace)
displace_out = wp.array((num_instances, num_instances_v*3), dtype=wp.float32, device="cuda:0")
wp.launch(reshape3, dim=num_instances, inputs=[displace], outputs=[displace_out], device="cuda:0")
print("displace_out:", displace_out)