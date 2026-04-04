import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from globals import *
from base_mesh import *
from instances import *
from step import *
import time
import os
from read_obj import create_compiled_f

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

base_mesh_name = globals.BASE_MESH_NAME
obj_path = OBJ_PATHS[base_mesh_name]
select = OBJ_SELECT_FACES[base_mesh_name]
base_mesh = create_basemesh(obj_path=obj_path, select_faces=select)

tet_name = globals.TET_NAME
tet_path = MSH_PATHS[tet_name]
pinned_vertices = PINNED_VERTICES[tet_name]
base_instance = create_base_instance(file_path=tet_path, n_modes=globals.N_MODES, pinned_vertices=pinned_vertices, scale=globals.INSTANCE_SCALE)
instances_object = create_instances_object(base_mesh, base_instance)

time_step = 0
time_step_size = globals.TIME_STEP_SIZE
run = False
step = False
mode = 0
# ps_meshes = [None] * (instances_object.num_instances + tear_instances_object.num_instances)
ps_meshes = [None] * (instances_object.num_instances)

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
        if time_step > 20:
            t = 20*time_step_size
        else: 
            t = time_step*time_step_size
            
        if globals.MOVE == "spin":
            c = np.cos(8*t)
            s = np.sin(8*t)

            R_y = np.array([
                [ c, 0.0,  s],
                [0.0, 1.0, 0.0],
                [-s, 0.0,  c]
            ], dtype=np.float32)
            spin_base = base_mesh.resting_v @ R_y.T

            bm_update_v(base_mesh, spin_base)
        elif globals.MOVE == "slam":
            R_y = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            displace_base = base_mesh.resting_v + np.array([0,0,0.5*np.sin(5*t)])
            bm_update_v(base_mesh, displace_base)
        
        ps_base_mesh = ps.register_surface_mesh("base mesh", base_mesh.v_cur, base_mesh.all_f)

        '''then update the instances'''
        wp_update_all_instances(base_mesh,base_instance,instances_object, R_y.T)
        # wp_update_all_instances(base_mesh,tear_base_instance,tear_instances_object)

        tets = base_instance.tets
        v_curs = instances_object.v_cur.numpy()
        for i in range(instances_object.num_instances):
            vertices = v_curs[i]
            if time_step == 1:
                # m = ps.register_volume_mesh("tet mesh" + str(i), vertices, tets=tets)
                m = ps.register_surface_mesh("instance mesh" + str(i), vertices, base_instance.boundary_f)
                ps_meshes[i] = m
            else:
                ps_meshes[i].update_vertex_positions(vertices)

    step = False

if OUTPUT_TYPE == "polyscope":
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
    if OUTPUT_TYPE == "usd":
        filetime = time.strftime("%Y%m%d-%H%M")
        job_id = os.getenv("SLURM_JOB_ID", "nojobid")
        fname = "out/" + base_mesh_name + "_" + str(NUM_FRAMES) + "_" + job_id + "_"+ filetime + ".usdc"
        w = USDMultiMeshWriter(fname, fps=24, stage_up="Z", mesh_up="Y", write_velocities=True)
        w.open()

        counts = np.full(base_mesh.all_f.shape[0], 3)
        indices = base_mesh.all_f.flatten()
        w.add_mesh("basemesh",  counts, indices, num_points=base_mesh.v_cur.shape[0])

        counts = np.full(base_instance.boundary_f.shape[0] * instances_object.num_instances, 3)
        compiled_f = create_compiled_f(base_instance.boundary_f, base_instance.boundary_v.shape[0], instances_object.num_instances)
        indices = compiled_f.flatten()
        w.add_mesh("instances", counts, indices, num_points=base_instance.boundary_v.shape[0]*instances_object.num_instances)

    elif OUTPUT_TYPE == "sequence":
        print("creating folders...")
        filetime = time.strftime("%Y%m%d-%H%M")
        job_id = os.getenv("SLURM_JOB_ID", "nojobid")
                #create enclosing folder
        enclosing_folder_name = "/scratch/thirty/yixinlok/" + base_mesh_name + "_" + job_id + "_"+ filetime 
        os.makedirs(enclosing_folder_name, exist_ok=True)
        # create hedgehog folder
        hedgehog_folder_name = enclosing_folder_name + "/" + base_mesh_name
        os.makedirs(hedgehog_folder_name, exist_ok=True)
        # create instance folders
        instance_folder_name = enclosing_folder_name + "/" + tet_name
        os.makedirs(instance_folder_name, exist_ok=True)
        print("folders created")


    start = time.time()
    # wp.timing_begin(cuda_filter=wp.TIMING_MEMCPY)
    with wp.ScopedTimer("update all", cuda_filter=wp.TIMING_ALL):
        for time_step in range(NUM_FRAMES):
            ''' update the base first'''
            if time_step > 20:
                t = 20*time_step_size
            else: 
                t = time_step*time_step_size
                
            if globals.MOVE == "spin":
                c = np.cos(8*t)
                s = np.sin(8*t)

                R_y = np.array([
                    [ c, 0.0,  s],
                    [0.0, 1.0, 0.0],
                    [-s, 0.0,  c]
                ], dtype=np.float32)
                spin_base = base_mesh.resting_v @ R_y.T
                bm_update_v(base_mesh, spin_base)
            elif globals.MOVE == "slam":
                R_y = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                displace = np.array([0,0,0.5*np.sin(5*t)])

                displace_base = base_mesh.resting_v + np.array([0,0,0.5*np.sin(5*t)])
                bm_update_v(base_mesh, displace_base)
            
            wp_update_all_instances(base_mesh,base_instance,instances_object, R_y.T)
            
            
            if OUTPUT_TYPE == "usd":
                w.write_points("basemesh", base_mesh.v_cur,  timecode=time_step)
                reshaped_vs = instances_object.v_cur.numpy().reshape(-1,3)
                w.write_points("instances", reshaped_vs, timecode=time_step)

            elif OUTPUT_TYPE == "sequence":
                print(f"writing frame {time_step}...")
                igl.writeOBJ(hedgehog_folder_name + "/frame_" + str(time_step) + ".obj", base_mesh.v_cur, base_mesh.all_f) 
                compiled_f = create_compiled_f(base_instance.boundary_f, base_instance.boundary_v.shape[0], instances_object.num_instances)
                reshaped_vs = instances_object.v_cur.numpy().reshape(-1,3)
                igl.writeOBJ(instance_folder_name + "/frame_" + str(time_step) + ".obj", reshaped_vs, compiled_f)
    end = time.time()
    elapsed = end - start
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME).print_stats(30)
    # results = wp.timing_end()
    # wp.timing_print(results)
    
    print("---------------------------------------------------------")
    print("->" + base_mesh_name + " with " + tet_name + " instances")
    print("Number of modes:", N_MODES)
    print("Move type:", globals.MOVE)
    print("Number of instances:", instances_object.num_instances)
    print("Number of vertices per instance:", base_instance.v.shape[0])
    print("Total number of vertices:", base_instance.v.shape[0]*instances_object.num_instances)
    print("Number of frames:", NUM_FRAMES)
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print("---------------------------------------------------------")
    if OUTPUT_TYPE == "usd":
        w.close()
    

    

