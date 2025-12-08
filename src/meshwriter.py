import numpy as np
from usdmultimeshwriter import USDMultiMeshWriter

w = USDMultiMeshWriter("out.usdc", fps=24, stage_up="Z", mesh_up="Y", write_velocities=True)
w.open()

# topology once (triangles)
F = np.array([[0,1,2],[2,1,3]], dtype=np.int32)
counts = np.array([3,3]); indices = F.flatten()
w.add_mesh("Left",  counts, indices, num_points=4)
w.add_mesh("Right", counts, indices, num_points=4)

baseL = np.array([[-1,0,0],[0,0,0],[-1,1,0],[0,1,0]], dtype=np.float32)
baseR = np.array([[ 0,0,0],[1,0,0],[ 0,1,0],[1,1,0]], dtype=np.float32)

for k in range(60):
    fall = -0.02 * k  # Y-down in sim
    VL = baseL.copy(); VL[:,1] += fall
    VR = baseR.copy(); VR[:,1] -= fall
    w.write_points("Left", VL,  timecode=k)
    w.write_points("Right", VR, timecode=k)

w.close()