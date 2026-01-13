import numpy as np
import torch
from dyrt_utils import *
import warp as wp
from base_instance import *
from base_mesh import *
from globals import *

class Instances:
    # storing all data related to placing instances on the base mesh
    def __init__(self, vertices_by_instance, face_indices, barycentric, n_modes, n_vertices):
        # instances = Instances()

        self.n_modes = n_modes
        self.num_vertices = n_vertices
        self.num_instances = len(face_indices)

        self.face_indices = wp.from_numpy(np.array(face_indices).astype(np.int32), device=DEVICE)
        self.barycentric = wp.from_numpy(np.array(barycentric).astype(np.float32), device=DEVICE)
        
        # Get torch device
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        # instance_vertices has shape(num_instances, num_vertices, 3)
        vertices_by_instance = np.array(vertices_by_instance)
        self.v_cur = wp.from_numpy(vertices_by_instance.copy().astype(np.float32), device=DEVICE)
        self.v_prev = wp.from_numpy(vertices_by_instance.copy().astype(np.float32), device=DEVICE)
        self.v_prev2 = wp.from_numpy(vertices_by_instance.copy().astype(np.float32), device=DEVICE)

        # instance_vertices has shape(num_instances, n_modes)
        self.q_cur = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)
        self.q_prev = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)
        self.q_prev2 = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)

        return 

    def instances_update_v(self, new_vs):
        @wp.kernel
        def wp_update_v(
            new_v: wp.array(dtype=wp.mat((self.num_vertices, 3), dtype=float)),
            v_cur: wp.array(dtype=wp.mat((self.num_vertices, 3), dtype=float)), 
            v_prev: wp.array(dtype=wp.mat((self.num_vertices, 3), dtype=float)), 
            v_prev2: wp.array(dtype=wp.mat((self.num_vertices, 3), dtype=float))):

            tid = wp.tid()
            v_prev2[tid] = v_prev[tid]
            v_prev[tid] = v_cur[tid]
            v_cur[tid] = new_v[tid]
        wp.launch(wp_update_v, dim=new_vs.shape[0], inputs=[new_vs], outputs=[self.v_cur, self.v_prev, self.v_prev2], device=DEVICE)

        # @wp.kernel
        # def wp_update_v(
        #     new_v: wp.array2d(dtype=wp.vec3),
        #     v_cur: wp.array2d(dtype=wp.vec3),
        #     v_prev: wp.array2d(dtype=wp.vec3),
        #     v_prev2: wp.array2d(dtype=wp.vec3)):

        #     i,j = wp.tid()
        #     v_prev2[i][j] = v_prev[i][j]
        #     v_prev[i][j] = v_cur[i][j]
        #     v_cur[i][j] = new_v[i][j]
        # wp.launch(wp_update_v, dim=new_vs.shape, inputs=[new_vs], outputs=[self.v_cur, self.v_prev, self.v_prev2], device=DEVICE)


    def instances_update_q(self, new_qs):

        @wp.kernel
        def wp_update_q(
            new_q: wp.array2d(dtype=float),
            q_cur: wp.array2d(dtype=float), 
            q_prev: wp.array2d(dtype=float), 
            q_prev2: wp.array2d(dtype=float)):
            
            i,j = wp.tid()
            q_prev2[i][j] = q_prev[i][j]
            q_prev[i][j] = q_cur[i][j]
            q_cur[i][j] = new_q[i][j]
        
        wp.launch(wp_update_q, dim=new_qs.shape, inputs=[new_qs], outputs=[self.q_cur, self.q_prev, self.q_prev2], device=DEVICE)


def create_instances_object(
        bm: BaseMesh,
        bi: InstanceBase
        ):
    
    face_indices = []
    barycentrics = []
    new_instances_v = []
    for i in range(bm.faces_display):
        
        normal = bm.n[i] 
        # normal = wp.vec3f(normal[0], normal[1], normal[2])
        # rot_matrix = rodrigues_rotation_matrix(normal) 
        rot_matrix = rotate_to_align_with_z(normal)
        # get face vertices
        for j in range(bm.num_instance_per_face):
            face_indices.append(i)
            b1, b2, b3 = get_barycentric()
            barycentrics.append(np.array([b1, b2, b3]))

            face_point = bm_get_face_point(bm, i, (b1, b2, b3))
            new_instance_v = bi.v @ rot_matrix.T + face_point
            new_instances_v.append(new_instance_v)
    
    num_instances = bi.v.shape[0]

    instances = Instances(new_instances_v, face_indices, barycentrics, bi.n_modes, num_instances)
    return instances