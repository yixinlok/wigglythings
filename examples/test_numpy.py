import numpy as np
import warp as wp

@wp.kernel
def compute():

    t = wp.tile_arrange(0.0, 1.0, 0.1, dtype=float)
    s = wp.tile_map(wp.sin, t)

    print(s)

wp.launch_tiled(compute, dim=[1], inputs=[], block_dim=16, device="cuda:0")