import warp as wp
import warp.sparse as wps

import numpy as np  
import torch 

@wp.kernel()
def loss(eigenvectors: wp.array(dtype=wp.mat((num_instances_v*3,num_modes), dtype=float)),
        q_cur: wp.array(dtype=wp.vec(length=num_modes, dtype=float)),
        displace: wp.array(dtype=wp.vec(length=num_instances_v*3, dtype=float)),
        displace_t: wp.array(dtype=wp.mat((num_instances_v,3), dtype=float))):
    tid = wp.tid()
    displace[tid] = eigenvectors[0]@q_cur[tid]
    displace_t[tid] = wp.types.matrix(displace[tid], shape=(num_instances_v,3))