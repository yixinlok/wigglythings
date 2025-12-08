import numpy as np

def edge_lengths(X, E):

    """
    Compute the length of edges of a mesh defined by nodes X and edges E.

    Parameters
    ----------
    X : (n, 3) array
        Nodes of the mesh
    E : (m, 2) array
        Edges of the mesh

    Returns
    -------
    L : (m, 1) array
        Length of edges
    """

    L = np.linalg.norm(X[E[:, 0], :] - X[E[:, 1], :], axis=1)
    return L