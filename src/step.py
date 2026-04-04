import numpy as np
import torch 
import warp as wp
import time

from dyrt_params import *
from rodrigues_rotation import *
from base_mesh import *
from instances import *
from globals import *

    
def wp_update_all_instances(bm, bi, ix, frame_i):
    
    if frame_i >= globals.MOVE_FRAMES:
        frame_i = globals.MOVE_FRAMES - 1

    num_boundary_v = bi.boundary_v.shape[0]
    EV_LENGTH = wp.constant(bi.boundary_v.shape[0]*3)
    NUM_MODES = wp.constant(bi.n_modes)
    # displaces can be migrated
    displaces = torch.zeros((ix.num_instances, num_boundary_v,3), dtype=torch.float32, device=DEVICE)
    @wp.kernel
    def wp_get_modal_displacement(
        eigenvectors: wp.array2d(dtype=wp.float32),  # bi.v.shape[0]*3,bi.n_modes
        q_cur: wp.array2d(dtype=wp.float32), # ix.num_instances, bi.n_modes
        displaces: wp.array3d(dtype=wp.float32), # ix.num_instances, bi.v.shape[0], 3
        ):
        i = wp.tid()
        evs = wp.tile_load(eigenvectors, shape=(EV_LENGTH, NUM_MODES))
        qs = wp.tile_load(q_cur,shape=(1, NUM_MODES), offset=(i,0))
        qs = wp.tile_transpose(qs)

        d = wp.tile_zeros(shape=(EV_LENGTH, 1), dtype=wp.float32)
        wp.tile_matmul(evs, qs, d)
        d = wp.tile_reshape(d, (3, -1))
        d = wp.tile_transpose(d)
        wp.tile_store(displaces[i], d)
    wp.launch_tiled(wp_get_modal_displacement, 
                    dim=ix.num_instances, 
                    inputs=[bi.boundary_eigenvectors, ix.q_cur], 
                    outputs=[displaces], 
                    block_dim=128, 
                    device=DEVICE)

    # print("Uq time", T2-T1)
    face_indices = ix.face_indices
    # can be migrated into instances
    rot_matrices_T_array3d = torch.zeros((ix.num_instances,3,3), dtype=torch.float32, device=DEVICE)
    
    @wp.kernel
    def wp_get_rot_transpose(
            instances_face_index: wp.array(dtype=wp.int32),
            bm_normal: wp.array(dtype=wp.vec3),
            rot_matrices_T_array3d: wp.array3d(dtype=wp.float32)
        ):
        i = wp.tid()
        # instance_i = instances[i] 
        face = instances_face_index[i]
        normal = bm_normal[face] 
        rot_matrix = rodrigues_rotation_matrix(normal)
        
        # transpose and store into a 3d array
        rot_matrices_T_array3d[i][0][0] = rot_matrix[0,0]
        rot_matrices_T_array3d[i][0][1] = rot_matrix[1,0]
        rot_matrices_T_array3d[i][0][2] = rot_matrix[2,0]
        rot_matrices_T_array3d[i][1][0] = rot_matrix[0,1]
        rot_matrices_T_array3d[i][1][1] = rot_matrix[1,1]
        rot_matrices_T_array3d[i][1][2] = rot_matrix[2,1]
        rot_matrices_T_array3d[i][2][0] = rot_matrix[0,2]
        rot_matrices_T_array3d[i][2][1] = rot_matrix[1,2]
        rot_matrices_T_array3d[i][2][2] = rot_matrix[2,2]  

    wp.launch(wp_get_rot_transpose, 
              dim=ix.num_instances, 
              inputs=[ix.face_indices, bm.n_frames[frame_i]], 
              outputs=[rot_matrices_T_array3d], 
              device=DEVICE)
     

    @wp.kernel
    def wp_get_face_points(
        face_indices: wp.array(dtype=wp.int32),
        barycentrics: wp.array(dtype=wp.vec3),
        bm_v_cur: wp.array(dtype=wp.vec3),
        bm_f: wp.array(dtype=wp.vec3l),
        face_points: wp.array3d(dtype=wp.float32)
        ):
        i, j = wp.tid()
        face_point = wp_get_single_face_point(barycentrics[i], face_indices[i], bm_v_cur, bm_f)
        # face_point is a vec3
        face_points[i][j][0] = face_point[0]
        face_points[i][j][1] = face_point[1]
        face_points[i][j][2] = face_point[2]
    wp.launch(wp_get_face_points, 
              dim=(ix.num_instances, num_boundary_v), 
              inputs=[ix.face_indices, ix.barycentric, bm.v_frames[frame_i], bm.f_wp], 
              outputs=[ix.face_points], 
              device=DEVICE)
    


    # T4 = time.time()
    @wp.kernel
    def wp_compute_new_spikes(
        modal_d: wp.array2d(dtype=wp.vec3), # ix.num_instances,bi.v.shape[0],3
        base_v: wp.array(dtype=wp.vec3), # bi.v.shape[0],3
        rot_matrices_T: wp.array(dtype = wp.types.matrix((3, 3), dtype=wp.float32)),
        face_points: wp.array2d(dtype=wp.vec3), # ix.num_instances,bi.v.shape[0],3
        base_rotate: wp.types.matrix((3, 3), dtype=wp.float32), # 3,3
        vs: wp.array2d(dtype=wp.vec3) # ix.num_instances,bi.v.shape[0],3
        ):
        
        i,j = wp.tid()
        new_v = base_v[j] + modal_d[i][j] # modal displace
        new_v = new_v @ base_rotate # base_rotate
        new_v = new_v @ rot_matrices_T[i] # face rotate
        new_v = new_v + face_points[i][j] # attach to triangle
        vs[i][j] = new_v

    wp.launch(wp_compute_new_spikes,
            dim=(ix.num_instances, num_boundary_v),
            inputs=[displaces, bi.boundary_v_wp, rot_matrices_T_array3d, ix.face_points, bm.R_frames[frame_i]],
            outputs=[ix.v_next],
            device=DEVICE)

    # T5 = time.time()
    # print("instances update time", T5-T4)

    ix.instances_update_v(ix.v_next)
    # call dyrt AFTER, compute next q based on current force
    # wp_dyrt(bm, bi, ix, frame_i)

    return ix


def wp_dyrt(bm, bi, ix, frame_i):
    if frame_i >= globals.MOVE_FRAMES + 2:
        frame_i = globals.MOVE_FRAMES + 1

    # directly create zero matrix
    instance_accelerations = wp.zeros((ix.num_instances), dtype=wp.vec3, device=DEVICE)
    @wp.kernel
    def wp_interpolate_acceleration(
        face_indices: wp.array(dtype=wp.int32),
        barycentric: wp.array(dtype=wp.vec3),
        faces: wp.array(dtype=wp.vec3l),
        fd_acceleration: wp.array(dtype=wp.vec3),
        instance_accelerations: wp.array(dtype=wp.vec3) # num_instances
        ):

        i = wp.tid()
        
        fi = face_indices[i]
        v1 = faces[fi][0]
        v2 = faces[fi][1]
        v3 = faces[fi][2]

        b1 = barycentric[i][0]
        b2 = barycentric[i][1]
        b3 = barycentric[i][2]

        instance_accelerations[i] = b1*fd_acceleration[v1] + b2*fd_acceleration[v2] + b3*fd_acceleration[v3]
    wp.launch(wp_interpolate_acceleration, 
            dim=ix.num_instances, 
            inputs=[ix.face_indices, ix.barycentric, bm.f_wp, bm.acceleration_frames[frame_i]], 
            outputs=[instance_accelerations], 
            device=DEVICE)


    # T6 = time.time()
    @wp.kernel
    def wp_get_third_terms(
        instance_accelerations: wp.array(dtype=wp.vec3), # ix.num_instances
        mtm: wp.types.matrix((bi.n_modes, 3), dtype=wp.float32), # bi.n_modes, 3
        third_terms: wp.array(dtype=wp.types.vector(length=bi.n_modes, dtype=wp.float32))  # ix.num_instances, num_modes
    ):
        i = wp.tid()
        third_terms[i] = mtm @ instance_accelerations[i]
    third_terms_2 = torch.zeros((ix.num_instances, bi.n_modes), dtype=torch.float32, device=DEVICE)
    wp.launch(wp_get_third_terms,
            dim=ix.num_instances,
            inputs=[instance_accelerations, bi.mtm],
            outputs=[third_terms_2],
            device=DEVICE)


    c1,c2,c3 = bi.IIR_params
    q = torch.zeros((ix.num_instances, bi.n_modes), dtype=torch.float32, device=DEVICE)
    @wp.kernel
    def get_q_new(
        c1: wp.types.vector(length=bi.n_modes, dtype=wp.float32),
        c2: wp.types.vector(length=bi.n_modes, dtype=wp.float32),
        c3: wp.types.vector(length=bi.n_modes, dtype=wp.float32),
        q_cur: wp.array2d(dtype=wp.float32),
        q_prev: wp.array2d(dtype=wp.float32),
        third_term: wp.array2d(dtype=wp.float32),
        q: wp.array2d(dtype=wp.float32)
        ):
        
        i,j = wp.tid()
        q[i][j] = c1[j]*q_cur[i][j] + c2[j]*q_prev[i][j] + c3[j]*third_term[i][j]

    wp.launch(get_q_new, 
                dim=(ix.num_instances, bi.n_modes),
                inputs=[c1,c2,c3, ix.q_cur, ix.q_prev, third_terms_2], 
                outputs=[q], 
                device=DEVICE)
    # T8 = time.time()
    # print("get new q time", T8-T7)

    ix.instances_update_q(q)
    return 