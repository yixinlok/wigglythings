import warp as wp

@wp.func
def normalize(v: wp.vec3) -> wp.vec3:
    return v / wp.length(v)

@wp.func
def rodrigues_rotation_matrix(v: wp.vec3) -> wp.mat33:
    u = wp.vec3(0.0, 1.0, 0.0)
    v = normalize(v)

    cross = wp.cross(u, v)
    dot = wp.dot(u, v)

    norm_cross = wp.length(cross)

    if norm_cross < 1e-8:
        # Parallel or opposite
        if dot > 0.0:
            return wp.mat33(1.0,0.0,0.0,
                            0.0,1.0,0.0,
                            0.0,0.0,1.0)
        else:
            # 180° rotation about any perpendicular axis
            axis = wp.vec3(1.0, 0.0, 0.0) if wp.abs(u[0]) < wp.abs(u[1]) else wp.vec3(0.0, 1.0, 0.0)
            k = normalize(wp.cross(axis, u))
            K = wp.mat33(0.0, -k[2], k[1],
                         k[2], 0.0, -k[0],
                         -k[1], k[0], 0.0)
            K2 = K @ K
            # R = I + 2K^2 for 180 degrees
            I = wp.mat33(1.0,0.0,0.0,
                         0.0,1.0,0.0,
                         0.0,0.0,1.0)
            return I + 2.0 * K2

    k = cross / norm_cross
    theta = wp.acos(wp.clamp(dot, -1.0, 1.0))

    s = wp.sin(theta)
    c = wp.cos(theta)

    K = wp.mat33(0.0, -k[2], k[1],
                 k[2], 0.0, -k[0],
                 -k[1], k[0], 0.0)

    K2 = K @ K
    I = wp.mat33(1.0,0.0,0.0,
                 0.0,1.0,0.0,
                 0.0,0.0,1.0)

    # Rodrigues formula: R = I + sinθ*K + (1 - cosθ)*K²
    R = I + s * K + (1.0 - c) * K2
    return R

@wp.kernel
def test_rodrigues(R: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    u = wp.vec3(0.0, 1.0, 0.0)    # y-axis
    v = wp.vec3(1.0, 2.0, 3.0)
    R[tid] = rodrigues_rotation_matrix(v)

@wp.func
def transpose33(M: wp.mat33) -> wp.mat33:
    return wp.mat33(
        M[0,0], M[1,0], M[2,0],
        M[0,1], M[1,1], M[2,1],
        M[0,2], M[1,2], M[2,2],
    )

if __name__ == "__main__":
    # Launch on GPU (or CPU)
    R = wp.zeros(1, dtype=wp.mat33)
    print(R)
    wp.launch(kernel=test_rodrigues, outputs=[R], dim=3, device=DEVICE)
    print(R)