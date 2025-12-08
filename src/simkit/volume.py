import igl

from simkit.edge_lengths import edge_lengths
def volume(V, F):
    """
    Compute the volume of a simplex defined with nodes V and faces F.

    Parameters
    ----------
    V : (n, 3) array
        Nodes of the mesh
    F : (m, 2|3|4) array
        Simpleces of the mesh (either edges, triangles or tets)
    """
    dim = V.shape[1]

    t = F.shape[1]

    if t == 2:
        vol = edge_lengths(V, F).reshape(-1, 1)
    if t == 3:
        vol = igl.doublearea(V, F).reshape(-1, 1) / 2
    elif t == 4:
        vol = igl.volume(V, F).reshape(-1, 1)
    else:
        ValueError("Only F.shape[1] == 2, 3 or 4 are supported")
    return vol