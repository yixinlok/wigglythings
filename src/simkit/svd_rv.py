import numpy as np
def svd_rv(F):
    '''
    Computes the rotation variant of the SVD for a list of n dxd matrices F,
    such that F = U @ S @ V.transpose(0, 2, 1) while ensuring that
    U @ V.transpose(0, 2, 1) is a rotation matrix.

    Follows F1 from https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf

    Parameters
    ----------
    F : (n, d, d) array
        List of square matrices

    Returns
    -------
    U : (n, d, d) array
        The the orthogonal singular vectors U
    S : (n, d, d) array
        The singular values
    V : (n, d, d) array
        The orthognal singular vectors V
    '''

    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)

    d = F.shape[1]
    [U, S, VT] = np.linalg.svd(F)

    V = VT.transpose([0, 2, 1])

    I = np.tile(np.identity(d), (F.shape[0], 1, 1))

    S = I * S[:, None, :]

    L = I
    L[:, d-1, d-1] = np.linalg.det(U @ V.transpose(0, 2, 1))

    detU = np.linalg.det(U)
    detV = np.linalg.det(V)
    uI = np.logical_and(detU < 0, detV > 0)[:, None, None]
    vI = np.logical_and(detV < 0, detU > 0)[:, None, None]

    Ut = uI * U @ L + np.logical_not(uI) * U
    Vt = vI * V @ L + np.logical_not(vI) * V
    St = S @ L

    return Ut, St, Vt