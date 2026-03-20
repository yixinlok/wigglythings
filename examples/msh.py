from pathlib import Path

import igl
import meshio
import numpy as np


root = Path(__file__).resolve().parents[1]
obj_path = root / "assets" / "scale.obj"
out_path = root / "assets" / "scale.msh"

# libigl reads OBJ robustly even when vt counts differ from vertex counts.
v, tc, n, f, ft, fn = igl.read_obj(str(obj_path))

points = np.asarray(v, dtype=np.float64)
triangles = np.asarray(f, dtype=np.int32)

if triangles.size == 0:
	raise ValueError(f"No triangle faces found in {obj_path}")

mesh = meshio.Mesh(points=points, cells=[("triangle", triangles)])
meshio.write(str(out_path), mesh)

print(f"Wrote {out_path}")