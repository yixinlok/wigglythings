import numpy as np
import gpytoolbox as gp
import polyscope as ps
import scipy as sp
from base_instance import *
from instance import *
import globals 
from dyrt_utils import *
from matrix_utils import *
from read_obj import read_obj
import warp as wp
from rodrigues_rotation import *

class BaseMesh:
    resting_v: np.ndarray
    v: dict
    f: np.ndarray #faces to render instances on
    all_f: np.ndarray # all faces
    n: np.ndarray
    instance_scale: float
    # instances: list[Instance]

    # consider moving out
    faces_display: int
    num_instance_per_face: int

def create_basemesh(
        vertices=None, 
        faces=None, 
        obj_path=None, 
        select_faces=None
    ) -> BaseMesh:

    bm = BaseMesh()
    # vertex positions will be continuously updated as we animate it
    if vertices is None or faces is None:
        v, f = read_obj(obj_path)
    else:
        v = vertices
        f = faces
    
    bm.v = {"cur": v.copy().astype(np.float32),
            "prev": v.copy().astype(np.float32),
            "prev2": v.copy().astype(np.float32)}
    
    '''
    bm.f is the faces we render instances on
    bm.all_f is all the faces in the base mesh
    '''
    bm.all_f = np.array(f).astype(np.int64)
    if select_faces is not None:
        bm.f = bm.all_f[select_faces]
    else:
        bm.f = bm.all_f

    bm.resting_v = v.copy()
    bm.n = gp.per_face_normals(bm.v["cur"],bm.f,unit_norm=True)
    
    # bm.v_cur = wp.from_numpy(v.copy(), dtype=wp.vec3)
    # bm.v_prev = wp.from_numpy(bm.v["prev"])
    # bm.v_prev2 = wp.from_numpy(bm.v["prev2"])
    
    bm.faces_display = 5
    # bm.faces_display = bm.f.shape[0]
    bm.num_instance_per_face = 1

    return bm

def bm_update_v(
        bm: BaseMesh, 
        new_v: np.ndarray
    ) ->None:
    bm.v["prev2"] = bm.v["prev"]
    bm.v["prev"] = bm.v["cur"]
    bm.v["cur"] = new_v.astype(np.float32)
    bm.n = gp.per_face_normals(bm.v["cur"],bm.f,unit_norm=True)

    # update the rest of the mesh too
    return

def bm_get_face_center(bm, face_idx):
    center = np.mean(bm.v["cur"][bm.f[face_idx]], axis=0)
    assert center.shape == (3,)
    return center

def bm_get_face_point(bm, face_idx, barycentric):
    v1, v2, v3 = bm.v["cur"][bm.f[face_idx]]
    b1, b2, b3 = barycentric
    face_point = b1*v1 + b2*v2 + b3*v3
    return face_point

@wp.func
def get_face_point(
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



def bm_fd_acceleration(bm):
    acceleration = (bm.v["cur"] - 2*bm.v["prev"] + bm.v["prev2"])/(globals.TIME_STEP_SIZE**2)
    return acceleration.astype(np.float32)



'''
testing stuff
'''
def face_picker(
        bm: BaseMesh
    ):
    '''
    Visualise a single instance, and pick pinned vertices if not provided
    If run is True, the instance will let you toggle eigenmodes
    If run is False, the instance will just let you pick pinned vertices

    This function contains polyscope and will not be a kernel or function
    '''
    ps.init()

    colours = np.array([[0, 0, 0] for _ in range(bm.all_f.shape[0])])
    picked = []
    def callback():
        nonlocal colours, picked

        
        mesh = ps.register_surface_mesh("mesh", bm.v["cur"], bm.all_f)
        
        mesh.set_selection_mode('faces_only')

        io = psim.GetIO()
        if io.MouseClicked[0]: # if clicked
            screen_coords = io.MousePos
            pick_result = ps.pick(screen_coords=screen_coords)
            print(pick_result)
            if(pick_result.is_hit and pick_result.structure_name == "mesh"):
                i = pick_result.structure_data["index"]
                print(f"picked face {pick_result.structure_data["index"]}")
                # add to pinned vertices
                if i not in picked:
                    picked.append(i)
                print(f"picked faces: {picked}")
                colours = np.array([[0, 0, 0] for _ in range(bm.all_f.shape[0])])
                colours[picked] = np.array([1, 0, 0])
        mesh.add_color_quantity("pinned", colours, enabled=True, defined_on='faces')

    ps.set_user_callback(callback)
    ps.set_autocenter_structures(False)
    
    ps.show()

if __name__ == "__main__":
    bm = create_basemesh(obj_path = globals.OBJ_PATHS["hedgehog"])
    face_picker(bm)