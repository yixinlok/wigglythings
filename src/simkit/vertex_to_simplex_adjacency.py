
from scipy.sparse import csc_matrix

def vertex_to_simplex_adjacency(T, nv):
    # Initialize empty lists to store row, column, and data for the triplet list
    rows, cols, data = [], [], []

    # Iterate through the tetrahedra and populate the triplet list
    for tetrahedron_idx, vertices in enumerate(T):
        for vertex in vertices:
            rows.append(vertex)
            cols.append(tetrahedron_idx)
            data.append(1)

    # Create a CSC sparse matrix from the triplet lists
    adjacency_matrix = csc_matrix((data, (rows, cols)), shape=(nv, len(T)), dtype=int)

    return adjacency_matrix