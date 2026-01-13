import numpy as np 
import scipy as sp
from scipy.spatial.transform import Rotation as R
import warp as wp

def normalize_by_row(arr):
    row_norms = np.linalg.norm(arr, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1  # avoid division by zero
    return arr / row_norms


def remove_rows(matrix, indices):
    """
    Remove rows at the specified indices from the matrix.
    matrix: np.ndarray, shape (n, m)
    indices: list or array of row indices to remove
    Returns: np.ndarray with specified rows removed
    """
    return np.delete(matrix, indices, axis=0)

def remove_columns(matrix, indices):
    """
    Remove columns at the specified indices from the matrix.
    matrix: np.ndarray, shape (n, m)
    indices: list or array of column indices to remove
    Returns: np.ndarray with specified columns removed
    """
    return np.delete(matrix, indices, axis=1)

def add_zeros_columns(matrix, indices):
    """
    Add zero columns back at the specified indices in the matrix.
    matrix: np.ndarray, shape (n, m)
    indices: list or array of column indices to add zeros
    Returns: np.ndarray with zero columns added at specified indices
    """
    total_cols = matrix.shape[1] + len(indices)
    new_matrix = np.zeros((matrix.shape[0], total_cols))
    current_col = 0
    indices_set = set(indices)
    
    for i in range(total_cols):
        if i in indices_set:
            new_matrix[:, i] = np.zeros(matrix.shape[0])
        else:
            new_matrix[:, i] = matrix[:, current_col]
            current_col += 1
            
    return new_matrix

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
        # indices_to_remove.extend([3 * i, 3 * i + 1, 3 * i + 2])
        indices_to_remove.extend([i, num_vertices + i, 2 * num_vertices + i])

    total_dofs = 3 * num_vertices
    all_indices = np.arange(total_dofs)

    selection_matrix = np.eye(total_dofs)
    selection_matrix = remove_columns(selection_matrix, indices_to_remove)
    selection_matrix = zero_out_rows(selection_matrix, indices_to_remove)
    selection_matrix = sp.sparse.csr_matrix(selection_matrix)
    return selection_matrix

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

def test_remove_columns():
    mat = np.random.randint(0, 10, size=(2, 5))
    print("original matrix",mat)
    new_mat = remove_columns(mat, [1, 3, 1])
    print("remove rows 1,3", new_mat)

def test_create_selection_matrix():
    print("test create selection matrix")
    print("2 vertices, pin vertex 1")
    mat = create_selection_matrix(2, [1])
    print(mat)

def test_add_zeros_columns():
    mat = np.random.randint(0, 10, size=(2, 3))
    print("original matrix",mat)
    new_mat = add_zeros_columns(mat, [1, 3])
    print("add zero columns at 1,3", new_mat)

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
    

if __name__ == "__main__":
    # test_add_zeros_rows()
    test_create_selection_matrix()
    

