
    # @wp.kernel()
    # def wp_get_modal_displacement(
    #     eigenvectors: wp.types.matrix((num_boundary_v*3,bi.n_modes), dtype=wp.float32),
    #     q_cur: wp.array(dtype = wp.types.vector(length=bi.n_modes, dtype=wp.float32)),
    #     out: wp.array(dtype = wp.types.vector(length=num_boundary_v*3, dtype=wp.float32))):

    #     tid = wp.tid()
    #     out[tid]= eigenvectors@q_cur[tid]

    # wp.launch(wp_get_modal_displacement, 
    #                 dim=ix.num_instances, 
    #                 inputs=[bi.boundary_eigenvectors, ix.q_cur], 
    #                 outputs=[displaces],  
    #                 device=DEVICE)

    # displaces = torch.reshape(displaces, (ix.num_instances, 3, -1))
    # displaces = displaces.transpose(1,2).contiguous()