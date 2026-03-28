import igl
import numpy as np
import os

v, f = igl.read_triangle_mesh("assets/hedgehog.obj")
# maybe use gpytoolbox write_mesh instead
# create a folder 
num_hedgehogs = 3
for h in range(num_hedgehogs):
    os.makedirs(f"out/test/hedgehog{h}", exist_ok=True)

for i in range(10):
    displace_base = v + np.array([0,0,0.5*np.sin(5*i)])
    igl.writeOBJ("out/test/hedgehog1/hedgehog_1_frame" + str(i) + ".obj", displace_base, f)
    displace_base = v + np.array([1,1,-0.5*np.sin(5*i)])
    igl.writeOBJ("out/test/hedgehog2/hedgehog_2_frame" + str(i) + ".obj", displace_base, f)
    displace_base = v + np.array([-1,-1,-0.5*np.sin(5*i)])
    igl.writeOBJ("out/test/hedgehog3/hedgehog_3_frame" + str(i) + ".obj", displace_base, f)
    
