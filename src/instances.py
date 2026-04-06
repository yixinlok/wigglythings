import numpy as np
import torch
from dyrt_params import *
import warp as wp
from base_instance import *
from base_mesh import *
from globals import *

class Instances:
    # storing all data related to placing instances on the base mesh
    def __init__(self, vertices_by_instance, face_indices, barycentric, n_modes, num_boundary_v):
        
        self.n_modes = n_modes
        self.num_instances = len(face_indices)

        self.face_indices = wp.from_numpy(np.array(face_indices).astype(np.int32), device=DEVICE)
        self.barycentric = wp.from_numpy(np.array(barycentric).astype(np.float32), device=DEVICE)
        
        
        # Get torch device
        self.torch_device = "cpu" if DEVICE == "cpu" else "cuda"
        # instance_vertices has shape(num_instances, num_vertices, 3)
        vertices_by_instance = torch.from_numpy(np.array(vertices_by_instance).astype(np.float32)).to(device=self.torch_device)        

        self.v_next = torch.zeros_like(vertices_by_instance)
        self.v_cur = wp.from_torch(vertices_by_instance.clone().to(device=self.torch_device))
        self.v_prev = wp.from_torch(vertices_by_instance.clone().to(device=self.torch_device))
        self.v_prev2 = wp.from_torch(vertices_by_instance.clone().to(device=self.torch_device))

         # instance_vertices has shape(num_instances, n_modes)
        self.q_cur = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)
        self.q_prev = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)
        self.q_prev2 = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)
        
        # allocate memory once, avoid runtime memset
        self.modal_displaces = torch.zeros((self.num_instances, num_boundary_v, 3), dtype=torch.float32, device=self.torch_device)
        self.face_points = torch.zeros((self.num_instances, num_boundary_v, 3), dtype=torch.float32, device=self.torch_device)
        self.rot_T = torch.zeros((self.num_instances, 3, 3), dtype=torch.float32, device=self.torch_device)
        self.accelerations = wp.zeros((self.num_instances), dtype=wp.vec3, device=DEVICE)
        self.third_terms = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)
        self.q_new = torch.zeros((self.num_instances, n_modes), dtype=torch.float32, device=self.torch_device)

    def instances_update_v(self, new_vs):
        @wp.kernel
        def wp_update_v(
            new_v: wp.array3d(dtype=wp.float32),
            v_cur: wp.array3d(dtype=wp.float32),
            v_prev: wp.array3d(dtype=wp.float32),
            v_prev2: wp.array3d(dtype=wp.float32)):

            i,j,k = wp.tid()
            v_prev2[i][j][k] = v_prev[i][j][k]
            v_prev[i][j][k] = v_cur[i][j][k]
            v_cur[i][j][k] = new_v[i][j][k]
        wp.launch(wp_update_v, 
                  dim=new_vs.shape, 
                  inputs=[new_vs], 
                  outputs=[self.v_cur, self.v_prev, self.v_prev2], 
                  device=DEVICE)

    def instances_update_q(self, new_qs):
        @wp.kernel
        def wp_update_q(
            new_q: wp.array2d(dtype=wp.float32),
            q_cur: wp.array2d(dtype=wp.float32), 
            q_prev: wp.array2d(dtype=wp.float32), 
            q_prev2: wp.array2d(dtype=wp.float32)):
            
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
        
        normal = bm.resting_n[i] 
        rot_matrix = rotate_to_align_with_z(normal)
        # get face vertices
        for j in range(bm.num_instance_per_face):
            face_indices.append(i)
            b1, b2, b3 = get_barycentric()
            barycentrics.append(np.array([b1, b2, b3]))

            face_point = bm.get_face_point(i, (b1, b2, b3))
            new_instance_v = bi.boundary_v @ rot_matrix.T + face_point
            new_instances_v.append(new_instance_v)
    
    instances = Instances(new_instances_v, face_indices, barycentrics, bi.n_modes, bi.boundary_v.shape[0])
    return instances