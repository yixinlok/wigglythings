import numpy as np
from dyrt_utils import *
from base_mesh import *
import warp as wp
from rodrigues_rotation import *
from instances import *
from globals import *
import time
import torch 

def wp_update_all_instances(
        bm: BaseMesh,
        bi: InstanceBase,
        ix: Instances
    ):
    T1 = time.time()

    EV_LENGTH = wp.constant(bi.v.shape[0]*3)
    NUM_MODES = wp.constant(bi.n_modes)
    displaces = wp.zeros((ix.num_instances, bi.v.shape[0],3), dtype=wp.float32, device=DEVICE)
    evs = wp.from_torch(bi.eigenvectors)
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
                    inputs=[evs, ix.q_cur], 
                    outputs=[displaces], 
                    block_dim=64, 
                    device=DEVICE)
    
    T2 = time.time()
    print("displaces time", T2-T1)
    
    face_indices = ix.face_indices
    bm_normals = wp.from_numpy(bm.n.astype(np.float32), dtype=wp.vec3, device=DEVICE)
    rot_matrices_T_array3d = wp.zeros((ix.num_instances,3,3), dtype=wp.float32, device=DEVICE)
    
    # this can be faster
    @wp.kernel
    def wp_get_rot_transpose(
            instances_face_index: wp.array(dtype=wp.int32),
            bm_normal: wp.array(dtype=wp.vec3),
            rot_matrices_T_array3d: wp.array3d(dtype=float)
        ):
        i = wp.tid()
        # instance_i = instances[i] 
        face = instances_face_index[i]
        normal = bm_normal[face] 
        rot_matrix = rodrigues_rotation_matrix(normal)
        
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
              inputs=[face_indices, bm_normals], 
              outputs=[rot_matrices_T_array3d], 
              device=DEVICE)
     
    T3 = time.time()
    print("rot transpose time", T3-T2)

    barycentric = ix.barycentric
    v_cur = wp.from_numpy(bm.v_cur, device=DEVICE)
    bm_f = wp.from_numpy(bm.f.astype(np.int32), dtype=wp.vec3l, device=DEVICE)

    face_points = wp.zeros((ix.num_instances, 3), dtype=wp.float32, device=DEVICE)

    @wp.kernel
    def wp_get_face_points(
        face_indices: wp.array(dtype=wp.int32),
        barycentrics: wp.array(dtype=wp.vec3),
        bm_v_cur: wp.array(dtype=wp.vec3),
        bm_f: wp.array(dtype=wp.vec3l),
        face_points: wp.array2d(dtype=float)
        ):
        i = wp.tid()
        face_point = get_face_point(barycentrics[i], face_indices[i], bm_v_cur, bm_f)
        # face_point is a vec3
        face_points[i][0] = face_point[0]
        face_points[i][1] = face_point[1]
        face_points[i][2] = face_point[2]
    wp.launch(wp_get_face_points, 
              dim=ix.num_instances, 
              inputs=[face_indices, barycentric, v_cur, bm_f], 
              outputs=[face_points], 
              device=DEVICE)
    
    # change it to wp torch in base instance
    base_v = wp.from_torch(torch.from_numpy(bi.v).to(dtype=torch.float32).to(DEVICE))
    vs = wp.zeros((ix.num_instances, bi.v.shape[0],3), dtype=wp.float32, device=DEVICE)
    BI_NUM_V = wp.constant(bi.v.shape[0])

    @wp.kernel
    def wp_better_get_total_displacement(
        num_v: int,
        displaces: wp.array3d(dtype=float), # ix.num_instances,bi.v.shape[0],3
        base_v: wp.array2d(dtype=float), # bi.v.shape[0],3
        rot_matrices_T: wp.array3d(dtype=float),# ix.num_instances,3,3
        face_points: wp.array2d(dtype=float), # ix.num_instances,3
        vs: wp.array3d(dtype=float) # ix.num_instances,bi.v.shape[0],3
        ):
        i,j = wp.tid()
        # displace = wp.tile_load(displaces[i], shape=(1, 3), offset=(j,0))
        # base = wp.tile_load(base_v, shape=(1, 3), offset=(j,0))
        displace = wp.tile_load(displaces[i], shape=(BI_NUM_V, 3), offset=(0,0))
        base = wp.tile_load(base_v, shape=(BI_NUM_V, 3), offset=(0,0))
        displaced_base_v = base + displace

        # new_v = wp.tile_zeros(shape=(1, 3), dtype=float)
        new_v = wp.tile_zeros(shape=(BI_NUM_V, 3), dtype=float)
        rot_mat = wp.tile_load(rot_matrices_T[i], shape=(3, 3), offset=(0, 0))
        wp.tile_matmul(displaced_base_v, rot_mat, new_v)

        face_point = wp.tile_load(face_points, shape=(1,3), offset=(i,0))
        face_point = wp.tile_broadcast(face_point, shape=(BI_NUM_V, 3))
        new_v = new_v + face_point
        wp.tile_store(vs[i], new_v, offset=(j,0))
    wp.launch_tiled(wp_better_get_total_displacement, 
                    # dim=(ix.num_instances, bi.v.shape[0]),
                    dim=(ix.num_instances),
                    inputs=[bi.v.shape[0], displaces, base_v, rot_matrices_T_array3d, face_points],
                    outputs=[vs],
                    block_dim=128,
                    device=DEVICE)
   
    T4 = time.time()
    print("total displacement time", T4-T3)

    ix.instances_update_v(vs)

    T5 = time.time()
    print("instances update time", T5-T4)

    # call dyrt AFTER, compute next q based on current force
    wp_dyrt(bm, bi, ix)

    return ix



def wp_dyrt(bm, bi, ix):
    
    # scaling_constant = bi.scale
    scaling_constant = 1.0
    # scaling_constant = 1.0 / len(bi.pinned_vertices)

    num_faces = bm.f.shape[0]  
    num_instances = ix.num_instances
    num_modes = bi.n_modes
    num_instance_vertices = bi.v.shape[0]
    
    c1,c2,c3 = np.array(bi.IIR_params)
    barycentric = ix.barycentric
    face_indices = ix.face_indices

    faces = wp.from_numpy(bm.f.astype(np.int32), dtype=wp.vec3l, device=DEVICE)
    fd_acceleration = wp.from_numpy(bm_fd_acceleration(bm), device=DEVICE)
    estimate_accelerations = wp.from_numpy(np.zeros((num_instances, 3)).astype(np.float32), device=DEVICE)
    
    @wp.kernel
    def wp_estimate_accelerations(
        face_indices: wp.array(dtype=wp.int32),
        barycentric: wp.array(dtype=wp.vec3),
        faces: wp.array(dtype=wp.vec3l),
        fd_acceleration: wp.array(dtype=wp.vec3),
        estimate_accelerations: wp.array(dtype=wp.vec3)):

        i = wp.tid()
        
        fi = face_indices[i]
        v1 = faces[fi][0]
        v2 = faces[fi][1]
        v3 = faces[fi][2]

        b1 = barycentric[i][0]
        b2 = barycentric[i][1]
        b3 = barycentric[i][2]
        estimate_accelerations[i] = b1*fd_acceleration[v1] + b2*fd_acceleration[v2] + b3*fd_acceleration[v3]
    wp.launch(wp_estimate_accelerations, dim=num_instances, 
            inputs=[face_indices, barycentric, faces, fd_acceleration], 
            outputs=[estimate_accelerations], 
            device=DEVICE)
    T6 = time.time()

    estimate_accelerations = estimate_accelerations.numpy()
    third_terms = []
    for i in range(ix.num_instances):
        estimate_acceleration = estimate_accelerations[i]

        '''
        this block of code only applies the forces to the pinned vertices
        '''
        forcing_term = np.zeros(bi.v.shape, dtype=np.float32)
        mask = np.zeros(bi.v.shape[0], dtype=bool)
        mask[bi.pinned_vertices] = True
        forcing_term[mask] = estimate_acceleration
        forcing_term = forcing_term.ravel().T

        third_term = scaling_constant*(bi.phi_inv @ forcing_term)

        third_terms.append(third_term)
    third_terms = np.array(third_terms, dtype=np.float32)
    third_terms = wp.from_numpy(third_terms, device=DEVICE)

    

    BI_NUM_V = wp.constant(bi.v.shape[0])
    
    # @wp.kernel
    # def wp_get_third_terms(
    #     pinned_vertices: wp.array(dtype=wp.int32), # len(bi.pinned_vertices)
    #     estimate_accelerations: wp.array(dtype=wp.vec3), # ix.num_instances,3
    #     phi_inv: wp.array2d(dtype=float), # bi.v.shape[0]*3, bi.n_modes
    #     third_terms: wp.array2d(dtype=float)  # ix.num_instances, num_modes
    # ):
    #     i, j = wp.tid()
    #     forcing_term = wp.tile_zeros(BI_NUM_V, 3)
    #     # for each pinned vertex, add estimate_acceleration as a tile
    #     wp.tile_reshape(forcing_term, (num_instance_vertices*3, 1))
    #     wp.transpose(forcing_term)


    T7 = time.time()
    print("get third terms time", T7-T6)

    # q = torch.zeros((num_instances, num_modes), dtype=torch.float32).to(ix.torch_device)
    # q = wp.from_torch(q)
    q = wp.zeros((num_instances, num_modes), dtype=wp.float32, device=DEVICE)

    q_cur = ix.q_cur
    q_prev = ix.q_prev
    @wp.kernel
    def get_q_new(
        c1: wp.vec(length=num_modes, dtype=float),
        c2: wp.vec(length=num_modes, dtype=float),
        c3: wp.vec(length=num_modes, dtype=float),
        q_cur: wp.array2d(dtype=float),
        q_prev: wp.array2d(dtype=float),
        third_term: wp.array(dtype=wp.vec(length=num_modes, dtype=float)),
        q: wp.array2d(dtype=float)
        ):
        
        i,j = wp.tid()
        q[i][j] = c1[j]*q_cur[i][j] + c2[j]*q_prev[i][j] + c3[j]*third_term[i][j]

    wp.launch(get_q_new, 
                dim=(num_instances, num_modes),
                inputs=[c1,c2,c3, q_cur, q_prev, third_terms], 
                outputs=[q], 
                device=DEVICE)
    
    T8 = time.time()
    print("get new q time", T8-T7)
    ix.instances_update_q(q)
    return 