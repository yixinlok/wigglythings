    # eigenvectors = wp.from_numpy(np.array([bi.eigenvectors.astype(np.float32)]), device=DEVICE)
    # d = wp.from_numpy(np.zeros((ix.num_instances, bi.v.shape[0]*3), dtype=np.float32), device=DEVICE)
    # displaces = wp.from_numpy(np.zeros((ix.num_instances, bi.v.shape[0], 3), dtype=np.float32), device=DEVICE)

    # @wp.kernel
    # def wp_get_modal_displacement(
    #     eigenvectors: wp.array(dtype=wp.mat((bi.v.shape[0]*3,bi.n_modes), dtype=float)),
    #     q_cur: wp.array(dtype=wp.vec(length=bi.n_modes, dtype=float)),
    #     num_v_per_instance: int,
    #     displaces: wp.array(dtype=wp.mat(shape=(bi.v.shape[0],3), dtype=float)),
    #     d: wp.array(dtype=wp.vec(length=bi.v.shape[0]*3, dtype=float))):
        
    #     tid = wp.tid()
    #     d[tid] = eigenvectors[0]@q_cur[tid]

    #     displaces[tid] = wp.matrix(d[tid], shape=(num_v_per_instance,3))

    #     # equivalent to reshape by vertex, then transpose
    #     for i in range(num_v_per_instance*3):
    #         displaces[tid][i//num_v_per_instance][i%3] = d[tid][i]


    # wp.launch(wp_get_modal_displacement, dim=ix.num_instances, inputs=[eigenvectors, ix.q_cur, bi.v.shape[0]], outputs=[displaces,d], device="cuda:0")
    # print("displaces shape:", d.shape)

