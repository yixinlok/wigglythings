import numpy as np
import gpytoolbox as gp
import polyscope as ps
from base_instance import *
import globals 
from dyrt_params import *
from matrix_utils import *
import warp as wp
from rodrigues_rotation import *

class BaseMesh:
    resting_v: np.ndarray # original rest vertex positions
    resting_n: np.ndarray # original rest per-face normals
    f: np.ndarray #faces to render instances on
    f_wp: wp.array(dtype=wp.vec3l) # faces in wp format
    all_f: np.ndarray # all faces

    v_frames: list # precomputed vertex positions for each frame, in wp format
    n_frames: list # precomputed per-face normals for each frame, in wp format
    R_frames: list # precomputed rotation matrices for each frame, in numpy format
    acceleration_frames: list # precomputed vertex accelerations for each frame, in wp format

    faces_display: int
    num_instance_per_face: int
    
    def get_v_cur(self, frame_i):
        if frame_i < globals.MOVE_FRAMES:
            return self.v_frames[frame_i].numpy()
        else:
            return self.v_frames[globals.MOVE_FRAMES-1].numpy()

    def get_face_point(self, face_idx, barycentric):
        v1, v2, v3 = self.resting_v[self.f[face_idx]]
        b1, b2, b3 = barycentric
        face_point = b1*v1 + b2*v2 + b3*v3
        return face_point

def create_basemesh(vertices=None, faces=None, obj_path=None, select_faces=None):

    bm = BaseMesh()
    # vertex positions will be continuously updated as we animate it
    if vertices is None or faces is None:
        print(f"reading obj from {obj_path} ...")
        v, f = gp.read_mesh(obj_path)
        v = gp.normalize_points(v)
    else:
        v = vertices
        f = faces

    bm.resting_v = v.copy().astype(np.float32)
    '''
    bm.f is the faces we render instances on
    bm.all_f is all the faces in the base mesh
    '''
    bm.all_f = np.array(f).astype(np.int64)
    if select_faces is not None:
        bm.f = bm.all_f[select_faces]
    else:
        bm.f = bm.all_f
    
    bm.f_wp = wp.from_numpy(bm.f.astype(np.int32), dtype=wp.vec3l, device=DEVICE)
    
    # bm.faces_display = 20
    bm.faces_display = bm.f.shape[0]
    bm.num_instance_per_face = globals.NUM_INSTANCE_PER_FACE

    bm.resting_n = gp.per_face_normals(bm.resting_v,bm.f,unit_norm=True)

    return bm

def precompute_bm_frames(bm, move_frames, move_name):
    print(f"precomputing {move_frames} base mesh frames ...")
    bm_v_frames = []
    bm_n_frames = []
    bm_acceleration_frames = []
    bm_R_frames = []

    bm_v_frames.append(bm.resting_v)
    bm_n_frames.append(gp.per_face_normals(bm.resting_v,bm.f,unit_norm=True).astype(np.float32))
    bm_acceleration_frames.append(np.zeros_like(bm.resting_v, dtype=np.float32))
    bm_R_frames.append(np.eye(3, dtype=np.float32))

    for i in range(1,move_frames):
        t = i*globals.TIME_STEP_SIZE

        if move_name == "spin":
            c = np.cos(8*t)
            s = np.sin(8*t)

            bm_R = np.array([
                [ c, 0.0,  s],
                [0.0, 1.0, 0.0],
                [-s, 0.0,  c]
            ], dtype=np.float32)

            displaces = np.array([0,0,0])
        
        elif move_name == "slam":
            bm_R = np.eye(3, dtype=np.float32)
            displaces = np.array([0,0,0.5*np.sin(5*t)], dtype=np.float32)

        frame_i_v = bm.resting_v @ bm_R.T + displaces

        # compute acceleration with finite difference, assuming uniform timestep
        if i >= 2:
            frame_i_acceleration = (frame_i_v - 2*bm_v_frames[i-1] + bm_v_frames[i-2])/(globals.TIME_STEP_SIZE**2)
        elif i == 1:
            frame_i_acceleration = (frame_i_v - bm_v_frames[0])/(globals.TIME_STEP_SIZE**2)

        bm_v_frames.append(frame_i_v)
        bm_n_frames.append(gp.per_face_normals(frame_i_v,bm.f,unit_norm=True).astype(np.float32))
        bm_R_frames.append(bm_R.T)
        bm_acceleration_frames.append(frame_i_acceleration)
    
    for i in range(2):
        # compute acceleration for the last 2 frames
        if i >= 2:
            frame_i_acceleration = (frame_i_v - 2*bm_v_frames[i-1] + bm_v_frames[i-2])/(globals.TIME_STEP_SIZE**2)
        elif i == 1:
            frame_i_acceleration = (frame_i_v - bm_v_frames[0])/(globals.TIME_STEP_SIZE**2)
        bm_acceleration_frames.append(frame_i_acceleration)

    torch_device = "cpu" if DEVICE == "cpu" else "cuda"

    for i in range(move_frames):
        bm_v_frames[i] = wp.from_numpy(bm_v_frames[i], dtype=wp.vec3, device=DEVICE)
        bm_n_frames[i] = wp.from_numpy(bm_n_frames[i], dtype=wp.vec3, device=DEVICE)
        bm_R_frames[i] = wp.mat33(bm_R_frames[i])
        bm_acceleration_frames[i] = wp.from_numpy(bm_acceleration_frames[i], dtype=wp.vec3, device=DEVICE)
    
    for i in range(move_frames, move_frames + 2):
        bm_acceleration_frames[i] = wp.from_numpy(bm_acceleration_frames[i], dtype=wp.vec3, device=DEVICE)

    bm.v_frames = bm_v_frames
    bm.n_frames = bm_n_frames
    bm.acceleration_frames = bm_acceleration_frames
    bm.R_frames = bm_R_frames
    return



@wp.func
def wp_get_single_face_point(
    barycentric: wp.vec3,
    face_idx: int,
    v_cur: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=wp.vec3l)) -> wp.vec3:

    v1 = v_cur[faces[face_idx][0]]
    v2 = v_cur[faces[face_idx][1]]
    v3 = v_cur[faces[face_idx][2]]
     
    b1 = barycentric[0]
    b2 = barycentric[1]
    b3 = barycentric[2]
    face_point = b1*v1 + b2*v2 + b3*v3
    return face_point


'''
testing stuff
'''
def face_picker(
        bm: BaseMesh,
        picked_faces 
    ):
    '''
    Visualise a single instance, and pick pinned vertices if not provided
    If run is True, the instance will let you toggle eigenmodes
    If run is False, the instance will just let you pick pinned vertices

    This function contains polyscope and will not be a kernel or function
    '''
    ps.init()

    colours = np.array([[0, 0, 0] for _ in range(bm.all_f.shape[0])])
    picked = picked_faces.copy()
    def callback():
        nonlocal colours, picked


        mesh = ps.register_surface_mesh("mesh", bm.v_cur, bm.all_f)

        mesh.set_selection_mode('faces_only')
        mesh.add_color_quantity("pinned", colours, enabled=True, defined_on='faces')

        io = psim.GetIO()
        if io.MouseClicked[0]: # if clicked
            screen_coords = io.MousePos
            pick_result = ps.pick(screen_coords=screen_coords)
            print(pick_result)
            if(pick_result.is_hit and pick_result.structure_name == "mesh"):
                i = pick_result.structure_data["index"]
                # print(f"picked face {pick_result.structure_data["index"]}")
                # add to pinned vertices
                if i not in picked:
                    picked.append(i)
                else:
                    picked.remove(i)
                print(f"picked faces: {picked}")
                colours = np.array([[0, 0, 0] for _ in range(bm.all_f.shape[0])])
                colours[picked] = np.array([1, 0, 0])

    ps.set_user_callback(callback)
    ps.set_autocenter_structures(False)
    
    ps.show()

if __name__ == "__main__":
    bm = create_basemesh(obj_path = globals.OBJ_PATHS["pangolin"])
    face_picker(bm, picked_faces=globals.OBJ_SELECT_FACES["pangolin"])