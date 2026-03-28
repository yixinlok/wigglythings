import gpytoolbox as gp
import numpy as np
import warp as wp

def read_obj(path):
    # while we don't have precomputed v and f to update, read from obj file
    print(f"reading obj from {path} ...")
    v, f = gp.read_mesh(path)
    v = gp.normalize_points(v)

    return v, f

def create_compiled_f(original_f, num_vertices, num_instances):
    # create compiled_f for later use in writing obj files
    compiled_f = []
    for i in range(num_instances):
        instance_i_f = []
        for face in original_f:
            instance_i_f.append([face[0]+i*num_vertices, face[1]+i*num_vertices, face[2]+i*num_vertices])
        compiled_f.extend(instance_i_f)
        
    return np.array(compiled_f)


# test compiled_f

def test_compiled_f():
    original_f = np.array([[0,1,2], [2,3,0]])
    num_vertices = 4
    num_instances = 3

    compiled_f = create_compiled_f(original_f, num_vertices, num_instances)
    print(compiled_f)

if __name__ == "__main__":
    test_compiled_f()