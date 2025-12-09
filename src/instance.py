import numpy as np
from dyrt_utils import *
import warp as wp

@wp.struct
class Instance:
    n_modes: int
    num_vertices: int
    face_index: int
    barycentric: wp.vec3

    v_cur: wp.array(dtype=wp.vec3)
    v_prev: wp.array(dtype=wp.vec3)
    v_prev2: wp.array(dtype=wp.vec3)

    q_cur: wp.array(dtype=float)
    q_prev: wp.array(dtype=float)
    q_prev2: wp.array(dtype=float)

def create_instance(
        v: np.ndarray,
        face_index: int,
        barycentric: np.ndarray,
        n_modes: int=6
    ) -> Instance:
    instance = Instance()

    instance.n_modes = n_modes
    instance.num_vertices = v.shape[0]
    instance.face_index = face_index
    instance.barycentric = wp.vec3((barycentric[0], barycentric[1], barycentric[2]))
    
    v = v.astype(np.float32)
    # length 3*num_vertices
    instance.v_cur = wp.from_numpy(v, dtype=wp.vec3, device=DEVICE)
    instance.v_prev = wp.from_numpy(v, dtype=wp.vec3, device=DEVICE)
    instance.v_prev2 = wp.from_numpy(v, dtype=wp.vec3, device=DEVICE)

    instance.q_cur = wp.zeros(n_modes, dtype=float, device=DEVICE)
    instance.q_prev = wp.zeros(n_modes, dtype=float, device=DEVICE)
    instance.q_prev2 = wp.zeros(n_modes, dtype=float, device=DEVICE)

    return instance


@wp.kernel
def wp_update_v(
    v_cur: wp.array(dtype=wp.vec3), 
    v_prev: wp.array(dtype=wp.vec3), 
    v_prev2: wp.array(dtype=wp.vec3), 
    new_v: wp.array(dtype=wp.vec3)):
    
    tid = wp.tid()
    v_prev2[tid] = v_prev[tid]
    v_prev[tid] = v_cur[tid]
    v_cur[tid] = new_v[tid]

@wp.kernel
def wp_update_v_instance(
    instance: Instance,
    new_v: wp.array(dtype=wp.vec3)
    ):
    tid = wp.tid()
    instance.v_prev2[tid] = instance.v_prev[tid]
    instance.v_prev[tid] = instance.v_cur[tid]
    instance.v_cur[tid] = new_v[tid]

@wp.kernel
def wp_update_q(
    q_cur: wp.array(dtype=float), 
    q_prev: wp.array(dtype=float), 
    q_prev2: wp.array(dtype=float), 
    new_q: wp.array(dtype=float)):
    
    tid = wp.tid()
    q_prev2[tid] = q_prev[tid]
    q_prev[tid] = q_cur[tid]
    q_cur[tid] = new_q[tid]

# def update_v(instance, new_v):
#     instance.v_prev2 = instance.v_prev
#     instance.v_prev = instance.v_cur
#     instance.v_cur = wp.from_numpy(new_v, dtype=wp.vec3, device=DEVICE)
#     return
    
# def update_q(instance, new_q):
#     instance.q_prev2 = instance.q_prev
#     instance.q_prev = instance.q_cur
#     instance.q_cur = wp.from_numpy(new_q, dtype=float, device=DEVICE)
#     return


if __name__ == "__main__":
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    i = Instance(v)
    i.dyrt(c=(1, 2, 3))
