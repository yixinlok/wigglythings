import polyscope as ps
import polyscope.imgui as psim
import gpytoolbox as gp
import numpy as np
import scipy as sp
from globals import *
from base_mesh import *
from instances import *
from step import *

import cProfile, pstats
import warp as wp
from usdmultimeshwriter import USDMultiMeshWriter

wp.config.quiet = False
wp.init()

if not wp.get_cuda_device_count():
    print(
        "Some snippets in this notebook assume the presence of "
        "a CUDA-compatible device and won't run correctly without one."
    )

base_mesh_name = "hedgehog"
obj_path = OBJ_PATHS[base_mesh_name]
select = OBJ_SELECT_FACES[base_mesh_name]
base_mesh = create_basemesh(obj_path=obj_path, select_faces=select)

tet_name = "spike"
tet_path = MSH_PATHS[tet_name]
pinned_vertices = PINNED_VERTICES[tet_name]
base_instance = create_base_instance(file_path=tet_path, n_modes=20, pinned_vertices=pinned_vertices, scale=0.1)
instances_object = create_instances_object(base_mesh, base_instance)

tear_tet_path = MSH_PATHS["teardrop"]
tear_pinned_vertices = PINNED_VERTICES["teardrop"]
tear_base_instance = create_base_instance(file_path=tear_tet_path, n_modes=20, pinned_vertices=tear_pinned_vertices, scale=0.1)
tear_instances_object = create_instances_object(base_mesh, tear_base_instance)

print("number of instances:", instances_object.num_instances)
print("number of vertices per instance:", base_instance.v.shape[0])
print("total number of vertices:", base_instance.v.shape[0]*instances_object.num_instances)

time_step = 0
time_step_size = 0.1
run = False
step = False
mode = 0
ps_meshes = [None] * (instances_object.num_instances + tear_instances_object.num_instances)

def callback():
    global time_step, run, mode, step
    global base_mesh, base_instance, instances_array, instances_object
    global ps_meshes
    
    if(psim.Button("Run")):
        run = True
        psim.SameLine()
    if(psim.Button("Step")):
        step = True
        psim.SameLine()
    if(psim.Button("Stop")):
        run = False

    if run or step:
        time_step += 1

        ''' update the base first'''
        if time_step > 10:
        #    displace_base = np.array([0.5*np.sin(5*10*time_step_size), 0, 0])
            t = 10*time_step_size
            displace_base = np.array([0,0,0.5*np.sin(5*10*time_step_size)])

        else: 
            # displace_base = np.array([0.5*np.sin(5*time_step*time_step_size), 0, 0])
            t = time_step*time_step_size
            

        c = np.cos(5*t)
        s = np.sin(5*t)

        R_y = np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c]
        ], dtype=np.float32)
        displace_base = np.array([0,0,0.5*np.sin(5*t)])

        new = base_mesh.resting_v @ R_y.T
        # displace_base = np.array([0, 0, 0])

        bm_update_v(base_mesh, new)
        ps_base_mesh = ps.register_surface_mesh("base mesh", base_mesh.v["cur"], base_mesh.all_f)

        '''then update the instances'''
        wp_update_all_instances(base_mesh,base_instance,instances_object)
        # wp_update_all_instances(base_mesh,tear_base_instance,tear_instances_object)

        tets = base_instance.tets
        for i in range(instances_object.num_instances):
            vertices = instances_object.v_cur[i]
            
            if time_step == 1:
                m = ps.register_volume_mesh("tet mesh" + str(i), vertices, tets=tets)
                ps_meshes[i] = m
            else:
                ps_meshes[i].update_vertex_positions(vertices)

        # for i in range(tear_instances_object.num_instances):
        #     vertices = tear_instances_object.v_cur[i]
            
        #     if time_step == 1:
        #         m = ps.register_volume_mesh("tear tet mesh" + str(i), vertices, tets=tear_base_instance.tets)
        #         ps_meshes[instances_object.num_instances + i] = m
        #     else:
        #         ps_meshes[instances_object.num_instances + i].update_vertex_positions(vertices)

    step = False

if POLYSCOPE_OR_USD == "polyscope":
    with cProfile.Profile() as pr:
        # === polyscope and UI === #
        ps.init()
        ps.set_user_callback(callback)
        ps.set_automatically_compute_scene_extents(True)
        ps.set_length_scale(1)
        ps.reset_camera_to_home_view()
        ps.show()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(30)

else:
    w = USDMultiMeshWriter("out/out.usdc", fps=24, stage_up="Z", mesh_up="Y", write_velocities=True)
    w.open()

    counts = np.full(base_mesh.all_f.shape[0], 3)
    indices = base_mesh.all_f.flatten()
    w.add_mesh("basemesh",  counts, indices, num_points=base_mesh.v["cur"].shape[0])

    for i in range(instances_object.num_instances):
        counts = np.full(base_instance.f.shape[0], 3)
        indices = base_instance.f.flatten()
        w.add_mesh("instance_" + str(i),  counts, indices, num_points=base_instance.v.shape[0])

    for time_step in range(50):
        ''' update the base first'''
        if time_step > 10:
        #    displace_base = np.array([0.5*np.sin(5*10*time_step_size), 0, 0])
            t = 10*time_step_size
            displace_base = np.array([0,0,0.5*np.sin(5*10*time_step_size)])

        else: 
            # displace_base = np.array([0.5*np.sin(5*time_step*time_step_size), 0, 0])
            t = time_step*time_step_size

        c = np.cos(5*t)
        s = np.sin(5*t)

        R_y = np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c]
        ], dtype=np.float32)

        displace_base = np.array([0,0,0.5*np.sin(5*t)])
        new = base_mesh.resting_v @ R_y.T
        bm_update_v(base_mesh, new)

        w.write_points("basemesh", base_mesh.v["cur"],  timecode=time_step)

        wp_update_all_instances(base_mesh,base_instance,instances_object)
        
        for i in range(instances_object.num_instances):
            vertices = instances_object.v_cur[i]
            w.write_points("instance_" + str(i), vertices, timecode=time_step)

    w.close()
