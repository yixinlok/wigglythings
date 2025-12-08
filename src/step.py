import numpy as np
from dyrt_utils import *
from base_mesh import *
from instance import *
import warp as wp
from rodrigues_rotation import *
from instances import *
from globals import *

def wp_update_all_instances(
        bm: BaseMesh,
        bi: InstanceBase,
        ix: Instances
    ):

    bm_normals = wp.from_numpy(bm.n.astype(np.float32), dtype=wp.vec3)
    rot_matrices_T= np.ones((ix.num_instances,3,3), dtype=np.float32)  
    @wp.kernel
    def wp_get_rot_transpose(
            instances_face_index: wp.array(dtype=wp.int32),
            bm_normal: wp.array(dtype=wp.vec3),
            rot_matrices: wp.array(dtype=wp.mat33)
        ):
        i = wp.tid()
        # instance_i = instances[i] 
        face = instances_face_index[i]
        normal = bm_normal[face] 
        rot_matrix = rodrigues_rotation_matrix(normal)
        rot_matrices[i] = transpose33(rot_matrix)
    wp.launch(wp_get_rot_transpose, dim=ix.num_instances, inputs=[ix.face_indices, bm_normals], outputs=[rot_matrices_T], device=DEVICE)
     
    displaces = []
    for i in range(ix.num_instances):
        displace = bi.eigenvectors @ ix.q_cur[i]
        displace = np.reshape(displace, (3, -1)).T
        displaces.append(displace)

    displaces = np.array(displaces, dtype=np.float32)
    faces_wp = wp.from_numpy(bm.f.astype(np.int32), dtype=wp.vec3l)
    
    @wp.kernel
    def wp_get_displacement(
        displace: wp.array(dtype=wp.mat(displaces[0].shape, dtype=float)),
        base_v: wp.array(dtype=wp.mat((bi.v.shape[0],3), dtype=float)),
        num_v: int,
        rot_matrices_T: wp.array(dtype=wp.mat33),
        face_indices: wp.array(dtype=wp.int32),
        barycentrics: wp.array(dtype=wp.vec3),
        bm_v_cur: wp.array(dtype=wp.vec3),
        bm_f: wp.array(dtype=wp.vec3l),
        vs: wp.array(dtype=wp.mat((bi.v.shape[0],3), dtype=float))
        ):

        i = wp.tid()
        new_v = base_v[0] + displace[i]
        new_v = new_v @ rot_matrices_T[i]

        for j in range(num_v):
            new_v[j] = new_v[j] + get_face_point(barycentrics[i], face_indices[i], bm_v_cur, bm_f)
        vs[i] = new_v
    vs = np.zeros((ix.num_instances, bi.v.shape[0],3)).astype(np.float32)
    wp.launch(wp_get_displacement, dim=ix.num_instances, inputs=[displaces, np.array([bi.v]), bi.v.shape[0], rot_matrices_T, ix.face_indices, ix.barycentric, bm.v["cur"], faces_wp], outputs=[vs], device=DEVICE)
    ix.instances_update_v(vs)

    # call dyrt AFTER, compute next q based on current force
    wp_dyrt(bm, bi, ix)

    return ix



def wp_dyrt(bm, bi, instances_object):
    
    # scaling_constant = bi.scale
    scaling_constant = 1.0
    # scaling_constant = 1.0 / len(bi.pinned_vertices)

    num_faces = bm.f.shape[0]  
    num_instances = instances_object.num_instances
    num_modes = bi.n_modes
    num_instance_vertices = bi.v.shape[0]
    
    c1,c2,c3 = np.array(bi.IIR_params)
    barycentric = instances_object.barycentric
    face_indices = instances_object.face_indices

    faces = wp.from_numpy(bm.f.astype(np.int32), dtype=wp.vec3l)
    fd_acceleration = bm_fd_acceleration(bm)
    estimate_accelerations = np.zeros((num_instances, 3)).astype(np.float32)

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

    wp.launch(wp_estimate_accelerations, dim=num_instances, inputs=[face_indices, barycentric, faces, fd_acceleration], outputs=[estimate_accelerations], device=DEVICE)

    third_terms = []
    for i in range(instances_object.num_instances):
        estimate_acceleration = estimate_accelerations[i]

        '''
        this block of code only applies the forces to the pinned vertices
        '''
        zeros = np.zeros_like(estimate_acceleration)
        forcing_term = np.tile(zeros, (bi.v.shape[0], 1))
        mask = np.zeros(bi.v.shape[0], dtype=bool)
        mask[bi.pinned_vertices] = True
        forcing_term[mask] = estimate_acceleration
        forcing_term = forcing_term.ravel().T

        # print("forcing_term", forcing_term)
        '''
        this line of code applies the forces to all vertices in the instance
        '''
        # forcing_term = np.tile(estimate_acceleration, bi.v.shape[0]).T

        # print("forcing_term", forcing_term.shape)
        third_term = scaling_constant*(bi.phi_inv @ forcing_term)
        third_terms.append(third_term)
    third_terms = np.array(third_terms, dtype=np.float32)
    
    @wp.kernel
    def get_q(
        c1: wp.vec(length=num_modes, dtype=float),
        c2: wp.vec(length=num_modes, dtype=float),
        c3: wp.vec(length=num_modes, dtype=float),
        q_cur: wp.array(dtype=wp.vec(length=num_modes, dtype=float)),
        q_prev: wp.array(dtype=wp.vec(length=num_modes, dtype=float)),
        third_term: wp.array(dtype=wp.vec(length=num_modes, dtype=float)),
        q: wp.array(dtype=wp.vec(length=num_modes , dtype=float))):
        
        i = wp.tid()
        # q[i] = c1*q_cur[i] + c2*q_prev[i] + c3*third_term[i]
        result = wp.vec(length=num_modes, dtype=float)
        for j in range(num_modes):
            result[j] = c1[j]*q_cur[i][j] + c2[j]*q_prev[i][j] + c3[j]*third_term[i][j]
        q[i] = result

    q = np.zeros((num_instances, num_modes), dtype=np.float32)
    wp.launch(get_q, dim=num_instances,inputs=[c1,c2,c3, instances_object.q_cur, instances_object.q_prev, third_terms], outputs=[q], device=DEVICE)
    instances_object.instances_update_q(q)
    return 