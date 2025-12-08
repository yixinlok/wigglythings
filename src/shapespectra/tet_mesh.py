import igl
import gpytoolbox as gp
import torch
import numpy as np
import polyscope as ps

# get tetrahedra mesh
# get boundary faces
# get signed distance
# only keep random points that are <= 0 signed distance

def get_shape(pts, code):
    # check if points ar

    file_path="assets/elephant.msh"
    v, f, tets, *rest = igl.readMSH(file_path)
    v = v * np.array([1, code, 1])
    v = gp.normalize_points(v) * 2
    f, rest1, rest2 = igl.boundary_facets(tets)
    
    signed_distance,ind,b = gp.signed_distance(np.array(pts), np.array(v), np.array(f), use_cpp=True)

    mask = signed_distance > 0   
    arr2_filtered = pts[~mask]

    # ps.init()
    # ps.register_point_cloud("cloud",arr2_filtered)
    # ps.show()

    return arr2_filtered


def sample3D(n):
    random_tensor = torch.rand(n, 3) - 0.5
    random_tensor = random_tensor * 2
    return random_tensor

samples = sample3D(100000).numpy()
inside_pts = get_shape(samples, 0.5)


