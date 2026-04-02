import numpy as np 
import scipy as sp
from scipy.spatial.transform import Rotation as R
import warp as wp


def add_zeros_rows(matrix, indices):
    """
    Add zero rows back at the specified indices in the matrix.
    matrix: np.ndarray, shape (n, m)
    indices: list or array of row indices to add zeros
    Returns: np.ndarray with zero rows added at specified indices
    """
    total_rows = matrix.shape[0] + len(indices)
    new_matrix = np.zeros((total_rows, matrix.shape[1]))
    current_row = 0
    indices_set = set(indices)
    
    for i in range(total_rows):
        if i in indices_set:
            new_matrix[i, :] = np.zeros(matrix.shape[1])
        else:
            new_matrix[i, :] = matrix[current_row, :]
            current_row += 1
            
    return new_matrix

def zero_out_rows(matrix, indices):
    """
    Zero out rows at the specified indices in the matrix.
    matrix: np.ndarray, shape (n, m)
    indices: list or array of row indices to zero out
    Returns: np.ndarray with specified rows zeroed out
    """
    new_matrix = matrix.copy()
    for i in indices:
        new_matrix[i, :] = np.zeros(matrix.shape[1])
    return new_matrix

def create_selection_matrix(num_vertices, pinned_vertices):
    if pinned_vertices is None or len(pinned_vertices) == 0:
        return np.eye(3 * num_vertices)

    # assert that all pinned_vertices indices are valid
    for v in pinned_vertices:
        if v < 0 or v >= num_vertices:
            raise ValueError(f"pinned vertex index {v} is out of bounds for number of vertices {num_vertices}")
        
    indices_to_remove = []
    for i in pinned_vertices:
        indices_to_remove.extend([i, num_vertices + i, 2 * num_vertices + i])

    total_dofs = 3 * num_vertices
    all_indices = np.arange(total_dofs)

    selection_matrix = np.eye(total_dofs)
    selection_matrix = np.delete(selection_matrix, indices_to_remove, axis=1)
    selection_matrix = zero_out_rows(selection_matrix, indices_to_remove)
    selection_matrix = sp.sparse.csr_matrix(selection_matrix)
    return selection_matrix

def remove_nonboundary_vertices_from_eigenvectors(eigenvectors, boundary_vertex_indices):
    """
    Remove rows corresponding to non-boundary vertices from the eigenvectors matrix.
    eigenvectors: np.ndarray, shape (3*num_vertices, n_modes)
    boundary_vertex_indices: list or array of vertex indices that are on the boundary
    Returns: np.ndarray with rows corresponding to non-boundary vertices removed
    """
    num_vertices = eigenvectors.shape[0] // 3
    all_vertex_indices = set(range(num_vertices))
    non_boundary_indices = list(all_vertex_indices - set(boundary_vertex_indices))

    # Each vertex corresponds to 3 rows in the eigenvectors matrix (x, y, z)
    rows_to_remove = []
    for idx in non_boundary_indices:
        rows_to_remove.extend([idx, num_vertices + idx, 2 * num_vertices + idx])

    ret = np.delete(eigenvectors, rows_to_remove, axis=0)
    return ret

def adjust_face_matrix_vertex_indices_for_boundary(faces, boundary_vertex_indices):
    """
    Adjust face indices to account for the fact that non-boundary vertices have been removed from the eigenvectors matrix.
    faces: np.ndarray, shape (num_faces, vertices_per_face)
    boundary_vertex_indices: list or array of vertex indices that are on the boundary
    Returns: np.ndarray with adjusted face vertex indices
    """
    vertex_index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(boundary_vertex_indices)}
    adjusted_faces = np.copy(faces)
    for i in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            old_vertex_index = faces[i, j]
            if old_vertex_index in vertex_index_mapping:
                adjusted_faces[i, j] = vertex_index_mapping[old_vertex_index]
            else:
                raise ValueError(f"Vertex index {old_vertex_index} in faces is not a boundary vertex.")

    return adjusted_faces

def rotate_to_align_with_z(b):
    assert b.shape == (3,)
    rot, _ = R.align_vectors([b], [[0, 1, 0]])
    Rmat = rot.as_matrix()
    return Rmat
 
def test_rotate_to_align_with_z():
    b1 = np.array([0, 0, 1])  # already aligned
    R1 = rotate_to_align_with_z(b1)
    assert np.allclose(R1, np.eye(3)), "Failed for b1"

    b2 = np.array([0, 1, 0])  # should rotate 90 degrees around x-axis
    R2 = rotate_to_align_with_z(b2)
    expected_R2 = R.from_euler('x', -90, degrees=True).as_matrix()
    assert np.allclose(R2, expected_R2), "Failed for b2"

    b3 = np.array([1, 0, 0])  # should rotate -90 degrees around y-axis
    R3 = rotate_to_align_with_z(b3)
    expected_R3 = R.from_euler('y', 90, degrees=True).as_matrix()
    assert np.allclose(R3, expected_R3), "Failed for b3"

    print("All tests passed!")

def test_create_selection_matrix():
    print("test create selection matrix")
    print("2 vertices, pin vertex 1")
    mat = create_selection_matrix(2, [1])
    print(mat)



def test_add_zeros_rows():
    mat = np.random.randint(0, 10, size=(3, 2))
    print("original matrix",mat)
    new_mat = add_zeros_rows(mat, [1, 3])
    print("add zero rows at 1,3", new_mat)

def test_zero_out_rows():
    mat = np.random.randint(0, 10, size=(3, 2))
    print("original matrix",mat)
    new_mat = zero_out_rows(mat, [1])
    print("zero out rows at 1", new_mat)

def get_barycentric():
    """
    Generate random barycentric coordinates (u, v, w) such that u + v + w = 1 and u, v, w >= 0
    """
    import numpy as np
    u = np.random.rand()
    v = np.random.rand()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    w = 1 - u - v
    return u,v,w

def create_projection_matrix(num_vertices, pinned_vertices):
    """
    Create a matrix to pre multiply with the eigenvectors, to assign accelerations to only the pinned vertices
    """
    s = np.zeros((num_vertices, 3, 3))
    for i in range(len(pinned_vertices)):
        idx = pinned_vertices[i]
        s[idx] = np.eye(3)
    s = s.reshape((3*num_vertices, 3))
    return s

def test_create_projection_matrix():
    print("test create projection matrix")
    print("2 vertices, pin vertex 1")
    mat = create_projection_matrix(5, [1, 3])
    print(mat) 

if __name__ == "__main__":
    # test_add_zeros_rows()
    # test_create_selection_matrix()
    test_create_projection_matrix()
    

