import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
import gpytoolbox as gp
import igl
import scipy as sp
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.arap_hessian import arap_hessian
from simkit.neo_hookian_hessian import neo_hookean_hessian
from simkit.mass_matrix import mass_matrix
from matrix_utils import *
from dyrt_utils import *
from globals import *
from dyrt_utils import *

# TODO: 
# make eigenvectors into wp.array


class InstanceBase:
    n_modes: int
    scale: float

    v: np.array
    f: np.array
    tets: np.array

    eigenvalues: np.array
    eigenvectors: np.array
    phi_inv: np.array
    big_gamma: np.array

    IIR_params: tuple

    # note: we don't need to store pinned vertices. we already used them to modify the eigenvector array
    # pinned_vertices: list

def create_base_instance(file_path, n_modes=6, pinned_vertices=[], scale=1.0):
    bi = InstanceBase()
    bi.n_modes = n_modes
    bi.scale = scale
    if not file_path.endswith(".msh"):
        raise ValueError("only .msh files are supported sorry :(")
    
    print(f"reading instance mesh from {file_path} ...")
    v, f, tets, *rest = igl.readMSH(file_path)
    f, *rest = igl.boundary_facets(tets)
    v = gp.normalize_points(v)
    v = scale*v + np.array([0,0.5*scale,0])
    
    # Get torch device
    torch_device =  "cpu"
    
    # now the vertices are all normalized, and centred sitting on top of the xy plane
    bi.v = v.astype(np.float32)
    bi.f = f.astype(np.int32)
    bi.tets = tets.astype(int)


    if pinned_vertices == []:
        # run pinned vertices picker if not provided
        print("no pinned vertices provided, running picker ...")
        visualise_single_instance(bi, run=False, pinned_vertices=pinned_vertices)
        print("picked pinned vertices:", pinned_vertices)
    bi.pinned_vertices = pinned_vertices
    eigenvalues, eigenvectors, phi_inv, big_gamma, M = precompute(v, tets, n_modes, scale, pinned_vertices)

    bi.eigenvalues = eigenvalues.astype(np.float32)
    bi.eigenvectors = torch.from_numpy(eigenvectors.astype(np.float32)).to(torch_device)
    # bi.big_gamma = wp.from_numpy(big_gamma.astype(np.float32), device=DEVICE)
    bi.phi_inv = phi_inv.astype(np.float32)
    bi.M = M

    c1, c2, c3 = compute_IIR_params(bi.eigenvalues**0.5)
    bi.IIR_params = (c1, c2, c3)
    return bi


def precompute(v,tets,n_modes,scale,pinned_vertices=[]):
    # H and M both store it as [x1, x2, ..., y1, y2, ..., z1, z2, ...]
    H = arap_hessian(v, tets) 
    M = mass_matrix(v, tets)
    M_one = 0 #TODO: modify this to give a single mass value
    M = sp.sparse.kron(M, sp.sparse.eye(3)).tocsr()
    assert H.shape[0] == M.shape[0] and H.shape[1] == M.shape[1], "H and M must be the same size"

    print("solving eigenvalue problem ...")
    # access the eigenvector of each mode with eigenvectors[:, mode]
    if pinned_vertices is not []:
        # construct selection matrix
        print("applying pinned vertices ...")
        P = create_selection_matrix(v.shape[0], pinned_vertices)
        # modify H and M
        H = P.T @ H @ P
        M = P.T @ M @ P

    eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(H, k=n_modes*2, M=M, which='LM', sigma=1)

    # normalize each column of eigenvectors
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i]) 

    # filter out negative eigenvalues and corresponding eigenvectors
    positive_idx = eigenvalues > 0
    eigenvalues = eigenvalues[positive_idx]
    eigenvectors = eigenvectors[:, positive_idx]

    eigenvalues = eigenvalues[0:n_modes]
    eigenvectors = eigenvectors[:, 0:n_modes]

    # assert all eigenvalues are positive
    assert np.all(eigenvalues > 0), "all eigenvalues must be positive"
    print("done solving eigenvalue problem.")
    assert n_modes == eigenvectors.shape[1]
    
    # add back zero rows for pinned vertices
    if pinned_vertices is not []:
        # each pinned vertex corresponds to 3 zero rows
        zero_rows = []
        for vertices in pinned_vertices:
            zero_rows.extend([vertices, v.shape[0] + vertices, 2*v.shape[0] + vertices])
        eigenvectors = add_zeros_rows(eigenvectors, zero_rows)
    
    assert eigenvectors.shape == (3*v.shape[0], n_modes)
    debug_modes(n_modes, eigenvectors, run=False)

    phi_inv = sp.linalg.pinv(eigenvectors)
    big_gamma = scale*construct_big_gamma(v.T)

    return eigenvalues, eigenvectors, phi_inv, big_gamma, M_one


'''
testing stuff
'''
def visualise_single_instance(
        bi: InstanceBase, 
        run:bool =True,
        pinned_vertices:list =[]
    ):
    '''
    Visualise a single instance, and pick pinned vertices if not provided
    If run is True, the instance will let you toggle eigenmodes
    If run is False, the instance will just let you pick pinned vertices

    This function contains polyscope and will not be a kernel or function
    '''
    ps.init()
    bi.colours = np.array([[0, 0, 0] for _ in range(bi.v.shape[0])])

    #  shifted_v = v
    time_step = 0
    time_step_size = 0.1
    mode = 0

    def callback():
        nonlocal time_step, time_step_size, mode
        
        for i in range(bi.n_modes):
            if(psim.Button(f"Toggle Mode {i}")):
                psim.SameLine()
                mode = i
            
        time_step += 1
        t = time_step*time_step_size
        
        if run == True:
            eigenvector = bi.eigenvectors[:, mode]
            displace = np.cos(time_step*time_step_size)*eigenvector
            displace = np.reshape(displace, (3, -1)).T
            
            mesh = ps.register_volume_mesh("tet mesh", bi.v + displace, tets=bi.tets)
        else:
            mesh = ps.register_volume_mesh("tet mesh", bi.v, tets=bi.tets)

            cloud = ps.register_point_cloud("cloud", bi.v)
            io = psim.GetIO()
            if io.MouseClicked[0]: # if clicked
                screen_coords = io.MousePos
                pick_result = ps.pick(screen_coords=screen_coords)
                if(pick_result.is_hit and pick_result.structure_name == "cloud"):
                    # print(f"picked vertex {pick_result.local_index} at position {pick_result.position}")
                    # add to pinned vertices
                    if pick_result.local_index not in pinned_vertices:
                        pinned_vertices.append(pick_result.local_index)
                    print(f"pinned vertices: {pinned_vertices}")
                    bi.colours = np.array([[0, 0, 0] for _ in range(bi.v.shape[0])])
                    bi.colours[pinned_vertices] = np.array([1, 0, 0])
            cloud.add_color_quantity("pinned", bi.colours, enabled=True)

    ps.set_user_callback(callback)
    ps.set_autocenter_structures(False)
    
    ps.show()

def debug_modes(n_modes, eigenvectors, run=True):
    '''
    Visualise the histogram of eigenvector values for each mode
    '''
    import matplotlib.pyplot as plt

    if run == False:
        return

    fig, axs = plt.subplots(n_modes, 1, figsize=(8, 6))
    showHistogram = False
    for mode in range(n_modes):
        axs[mode].hist(eigenvectors[:, mode], bins=100, log=False)
        axs[mode].set_title(f"Histogram of eigenvector values, mode {mode}")
        axs[mode].set_xlabel("Value")
        axs[mode].set_ylabel("Frequency")      
        # plt.figtext(0.5, 0.01, f"min: {np.min(vec_norm)}, max: {np.max(vec_norm)}", ha="center", fontsize=10) 
    if showHistogram:
        plt.tight_layout()
        plt.show()   
    return 

if __name__ == "__main__":
    # for leaf, pin vertex 151
    # instance = Instance(mesh_path="assets/single_leaf.msh", n_modes=6, pinned_vertices=[151])
    # instance.visualise_single_instance()

    bi = create_base_instance(file_path="assets/feather.msh", n_modes=6, scale=0.3)
    visualise_single_instance(bi)
    # visualise_single_instance(bi, pinned_vertices=PINNED_VERTICES["spring"])


