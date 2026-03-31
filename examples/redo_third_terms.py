import warp as wp
import torch

num_instances = 2
v_shape = 4
n_modes = 3

forcing_term = wp.zeros((num_instances, bi.v.shape[0], 3), dtype=wp.float32, device=DEVICE)
@wp.kernel
def wp_get_forcing_terms(
    pinned_vertices: wp.array(dtype=wp.int32), # len(bi.pinned_vertices)
    estimate_accelerations: wp.array(dtype=wp.types.vector), # ix.num_instances
    forcing_term: wp.array2d(dtype=wp.float32) # ix.num_instances, bi.v.shape[0], 3
):
    i,j = wp.tid()
    
    est_acc = estimate_accelerations[i]
    vertex_idx = pinned_vertices[j]
    
    forcing_term[i][vertex_idx][0] = est_acc[0]
    forcing_term[i][vertex_idx][1] = est_acc[1]
    forcing_term[i][vertex_idx][2] = est_acc[2]

forcing_term = forcing_term.transpose(1,2) # ix.num_instances, 3, bi.v.shape[0]
forcing_term = torch.reshape(forcing_term, (num_instances, v_shape*3)) # ix.num_instances, bi.v.shape[0]*3

@wp.kernel
def wp_get_third_terms(
    forcing_term: wp.array(wp.types.vector(length=v_shape*3, dtype=wp.float32)), # ix.num_instances, bi.v.shape[0]*3
    p_inv: wp.types.matrix((n_modes, v_shape*3), dtype=wp.float32), # bi.n_modes, bi.v.shape[0]*3
    third_terms: wp.array(dtype=wp.types.matrix((num_instances, n_modes), dtype=wp.float32))  # ix.num_instances, num_modes
):
    i = wp.tid()
    third_terms[i] = p_inv @ forcing_term[i]
