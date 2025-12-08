
import numpy as np
import warp as wp
n=1


@wp.kernel
def saxpy(x: wp.array(dtype=wp.vec3), y: wp.mat((3,3), dtype=float), a: float):
    i = wp.tid()
    x[i] = y @ wp.vec3(5.0, 5.0, 5.0) 
    # y[i][0] = a * x[i][0] + y[i][0]

x = np.ones((5,3), dtype=np.float32)
y = np.ones((3,3), dtype=np.float32)

# y = np.ones((5,3), dtype=np.float32)
z = np.ones((3,1), dtype=np.float32)


wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device="cpu")
print(x)

x= np.ones((5,4,3), dtype=np.float32)   
y= np.zeros((4,3), dtype=np.float32)   
# y: wp.array(dtype=wp.mat((4,3), dtype=float))
shape = (4,3)
@wp.kernel
def getrot(x: wp.array(dtype=wp.mat(shape, dtype=float)) ,y: wp.mat(shape, dtype=float)):
    
    i = wp.tid()
    
    y = wp.mat(
        0.0, 1.0, 2.0,
        3.0, 4.0, 5.0,  
        6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, shape=(4,3)
    )
    x[i] = x[i] + y  # element-wise matrix addition
# inputs =[y], 
wp.launch(getrot, dim=5, inputs=[x, y], device="cpu")
print(x)
# print(y)


