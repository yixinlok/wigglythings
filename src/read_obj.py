import gpytoolbox as gp

def read_obj(path):
    # while we don't have precomputed v and f to update, read from obj file
    print(f"reading obj from {path} ...")
    v, f = gp.read_mesh(path)
    v = gp.normalize_points(v)

    return v, f