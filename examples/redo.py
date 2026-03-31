import warp as wp
import numpy as np
import torch 

num_instances = 2
num_v = 2
num_modes = 2


@wp.kernel()
def wp_get_modal_displacement(
        eigenvectors: wp.types.matrix((num_v*3,num_modes), dtype=wp.float32),
        q_cur: wp.array(dtype = wp.types.vector(length=num_modes, dtype=wp.float32)),
        out: wp.array(dtype = wp.types.vector((num_v*3), dtype=wp.float32))):
    tid = wp.tid()
    out[tid]= eigenvectors@q_cur[tid]

ev = torch.randn((num_v*3,num_modes), dtype=torch.float32).cuda()
q_cur = torch.ones((num_instances, num_modes), dtype=torch.float32).cuda()
md = torch.zeros((num_instances, num_v*3), dtype=torch.float32).cuda()

wp.launch(wp_get_modal_displacement, dim=num_instances, inputs=[ev, q_cur], outputs=[md], device="cuda:0")

md = torch.reshape(md, (num_instances, 3, -1))
md = md.transpose(1,2)
print("out:", md )


@wp.kernel
def is_instance_typecast_possible(
    md: wp.array3d(dtype=wp.float32), # num_instances, num_v*3
    out: wp.array3d(dtype=wp.float32)):
    i,j,k = wp.tid()
    out[i][j][k] = md[i][j][k]

out_md = torch.zeros((num_instances, 3, num_v), dtype=torch.float32).cuda()
wp.launch(is_instance_typecast_possible, dim=(num_instances, 3, num_v), inputs=[md], outputs=[out_md], device="cuda:0")
print("out_md:", out_md)
'''
conclusion: it is typecastable instantly
'''




@wp.kernel
def wp_compute_new_spikes(
    modal_d: wp.array(dtype = wp.types.matrix((num_v, 3), dtype=wp.float32)),
    base_v: wp.types.matrix((num_v, 3), dtype=wp.float32),
    rot_matrices_T: wp.array(dtype = wp.types.matrix((3, 3), dtype=wp.float32)),
    face_points: wp.array2d(dtype = wp.types.matrix((num_v, 3), dtype=wp.float32)), # ix.num_instances,3
    base_rotate: wp.types.matrix((3, 3), dtype=wp.float32), # 3,3
    vs: wp.array3d(dtype=wp.float32), # ix.num_instances,bi.v.shape[0],3
    ):
    i = wp.tid()

    new_v = base_v + modal_d[i] # modal displace
    new_v = new_v @ base_rotate # base_rotate
    new_v = new_v @ rot_matrices_T[i] # face rotate
    new_v = new_v + face_points[i] # attach to triangle



    