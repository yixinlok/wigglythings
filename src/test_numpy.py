import numpy as np

matrix = np.random.rand(12,1).astype(np.float32)

reshaped = matrix.reshape((3,4))

reshaped2 = matrix.reshape((4,3)).T 

# compare reshaped and reshaped2
print("reshaped:\n", reshaped)
print("reshaped2:\n", reshaped2)