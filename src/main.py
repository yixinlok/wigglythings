import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import warp as wp
import time, os, sys
import cProfile, pstats

from globals import *
from base_mesh import *
from base_instance import *
from instances import *
from step import *
from matrix_utils import *


wp.config.quiet = False
wp.init()

'''
setting up the base mesh (bm), base instance (bi) and instances object (ix)
'''
base_mesh_name = globals.BASE_MESH_NAME
obj_path = OBJ_PATHS[base_mesh_name]
select = OBJ_SELECT_FACES[base_mesh_name]
bm = create_basemesh(obj_path=obj_path, select_faces=select)

tet_name = globals.TET_NAME
tet_path = MSH_PATHS[tet_name]
pinned_vertices = PINNED_VERTICES[tet_name]
bi = create_base_instance(file_path=tet_path, n_modes=globals.N_MODES, pinned_vertices=pinned_vertices, scale=globals.INSTANCE_SCALE)
ix = create_instances_object(bm, bi)

'''
main algorithm
'''
precompute_bm_frames(bm, globals.MOVE_FRAMES, globals.MOVE)
def advance_frame(frame_i):
    wp_update_all_instances(bm, bi, ix, frame_i)
    wp_dyrt(bm, bi, ix, frame_i)
    

print("---------------------------------------------------------")
print("->" + base_mesh_name + " with " + tet_name + " instances")
print("Number of modes:", globals.N_MODES)
print("Move type:", globals.MOVE)
print("Number of instances:", ix.num_instances)
print("Number of vertices per instance:", bi.v.shape[0])
print("Total number of vertices:", bi.v.shape[0]*ix.num_instances)
print("Number of frames:", globals.NUM_FRAMES)
print("---------------------------------------------------------")

if OUTPUT_TYPE == "time":
    start = time.time()
    with wp.ScopedTimer("update all", cuda_filter=wp.TIMING_ALL):
        for frame_i in range(NUM_FRAMES):
            advance_frame(frame_i)
    end = time.time()
    elapsed = end - start
    print(f"Total time for {NUM_FRAMES} frames: {elapsed:.4f} seconds")
    sys.exit()

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

    compiled_f = create_compiled_f(bi.boundary_f, bi.boundary_v.shape[0], ix.num_instances)
    for frame_i in range(NUM_FRAMES):
        advance_frame(frame_i)

        print(f"writing frame {frame_i}...")
        igl.writeOBJ(hedgehog_folder_name + "/frame_" + str(frame_i) + ".obj", bm.get_v_cur(frame_i), bm.all_f) 
        reshaped_vs = ix.v_cur.numpy().reshape(-1,3)
        igl.writeOBJ(instance_folder_name + "/frame_" + str(frame_i) + ".obj", reshaped_vs, compiled_f)
    sys.exit()


'''
polyscope visualization and UI
'''              
ps_state = {
    "time_step": 0,
    "run": False,
    "step": False,
    "ps_base_mesh": None,
    "ps_meshes": []
}

def make_callback(state, bm, bi, ix):
    def callback():
        if(psim.Button("Run")):
            state.run = True
            psim.SameLine()
        if(psim.Button("Step")):
            state.step = True
            psim.SameLine()
        if(psim.Button("Stop")):
            state.run = False

        if state.run or state.step:
            advance_frame(state.time_step)

            v_curs = ix.v_cur.numpy()
            for i in range(ix.num_instances):
                vertices = v_curs[i]
                if state.time_step == 0:
                    ps_state["ps_base_mesh"] = ps.register_surface_mesh("base mesh", base_mesh.v_cur, base_mesh.all_f)
                    m = ps.register_surface_mesh("instance mesh" + str(i), vertices, bi.boundary_f)
                    ps_state["ps_meshes"].append(m)
                else:
                    ps_state["ps_base_mesh"].update_vertex_positions(base_mesh.v_cur)
                    ps_state["ps_meshes"][i].update_vertex_positions(vertices)
            state.time_step += 1
        state.step = False

if OUTPUT_TYPE == "polyscope":
    with cProfile.Profile() as pr:
        # === polyscope and UI === #
        ps.init()
        ps.set_user_callback(callback(ps_state, bm, bi, ix))
        ps.set_automatically_compute_scene_extents(True)
        ps.set_length_scale(1)
        ps.reset_camera_to_home_view()
        ps.show()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(30)