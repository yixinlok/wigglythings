# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Viewer
#
# Shows how to use the Newton Viewer class to visualize various shapes
# and line instances without a Newton model.
#
# Command: python -m newton.examples basic_viewer
#
###########################################################################

import math

import numpy as np
import warp as wp
# from pxr import Usd, UsdGeom

import newton
import newton.examples

import polyscope as ps
import polyscope.imgui as psim
import gpytoolbox as gp
import numpy as np
import scipy as sp
from globals import *
from base_mesh import *
from instances import *
from step import *



class Example:
    def __init__(self, viewer):
    
        self.viewer = viewer

        obj_path = OBJ_PATHS["hedgehog"]
        select = OBJ_SELECT_FACES["hedgehog"]
        tet_path = MSH_PATHS["loosecoil"]
        pinned_vertices = PINNED_VERTICES["loosecoil"]

        base_mesh = create_basemesh(obj_path=obj_path, select_faces=select)
        base_instance = create_base_instance(file_path=tet_path, n_modes=20, pinned_vertices=pinned_vertices, scale=0.1)
        instances_object = create_instances_object(base_mesh, base_instance)

        self.base_mesh = base_mesh
        self.base_instance = base_instance
        self.instances_object = instances_object

        self.col_bunny = wp.array([wp.vec3(0.5, 0.2, 0.8)], dtype=wp.vec3)
        self.col_plane = wp.array([wp.vec3(0.125, 0.125, 0.15)], dtype=wp.vec3)

        self.mat_default = wp.array([wp.vec4(0.0, 0.7, 0.0, 0.0)], dtype=wp.vec4)
        self.mat_plane = wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4)

        n= 100
        mesh_vertices = np.random.rand(n, 3).astype(np.float32)
        mesh_indices = np.random.randint(0, 100, size=(n,4))
        # mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        self.bunny_mesh = newton.Mesh(mesh_vertices, mesh_indices)
        self.bunny_mesh.finalize()

        # Demonstrate log_lines() with animated debug/visualization lines
        axis_eps = 0.01
        axis_length = 2.0
        self.axes_begins = wp.array(
            [
                wp.vec3(0.0, 0.0, axis_eps),  # X axis start
                wp.vec3(0.0, 0.0, axis_eps),  # Y axis start
                wp.vec3(0.0, 0.0, axis_eps),  # Z axis start
            ],
            dtype=wp.vec3,
        )

        self.axes_ends = wp.array(
            [
                wp.vec3(axis_length, 0.0, axis_eps),  # X axis end
                wp.vec3(0.0, axis_length, axis_eps),  # Y axis end
                wp.vec3(0.0, 0.0, axis_length + axis_eps),  # Z axis end
            ],
            dtype=wp.vec3,
        )

        self.axes_colors = wp.array(
            [
                wp.vec3(1.0, 0.0, 0.0),  # Red X
                wp.vec3(0.0, 1.0, 0.0),  # Green Y
                wp.vec3(0.0, 0.0, 1.0),  # Blue Z
            ],
            dtype=wp.vec3,
        )

        self.time = 0.0
        self.spacing = 2.0

    def gui(self, ui):
        ui.text("Custom UI text")
        _changed, self.time = ui.slider_float("Time", self.time, 0.0, 100.0)
        _changed, self.spacing = ui.slider_float("Spacing", self.spacing, 0.0, 10.0)

    def step(self):
        pass

    def render(self):
        # Begin frame with time
        self.viewer.begin_frame(self.time)

        # Clean layout: arrange objects in a line along X-axis
        # All objects at same height to avoid ground intersection
        base_height = 2.0
        base_left = 0.0

        # Simple rotation animations
        qy_slow = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3 * self.time)
        qx_slow = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2 * self.time)
        qz_slow = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4 * self.time)

        # Bunny: spinning at y = 6
        x_bunny_anim = wp.array([wp.transform([0.0, base_left, base_height], qz_slow)], dtype=wp.transform)
        base_left += self.spacing

        # Update instances via log_shapes

        self.viewer.log_shapes(
            "/bunny_instance",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            x_bunny_anim,
            self.col_bunny,
            self.mat_default,
            geo_src=self.bunny_mesh,
        )

        self.viewer.log_shapes(
            "/plane_instance",
            newton.GeoType.PLANE,
            (50.0, 50.0),
            wp.array([wp.transform_identity()], dtype=wp.transform),
            self.col_plane,
            self.mat_plane,
        )

        self.viewer.log_lines("/coordinate_axes", self.axes_begins, self.axes_ends, self.axes_colors)


        self.viewer.log_shapes(
            "/base_mesh",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            x_bunny_anim,
            self.col_bunny,
            self.mat_default,
            geo_src=self.bunny_mesh,
        )

        # End frame (process events, render, present)
        self.viewer.end_frame()

        self.time += 1.0 / 60.0

    def test(self):
        pass


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer)
    print(args)
    newton.examples.run(example, args)

# def init(parser=None):
#     """Initialize Newton example components from parsed arguments.

#     Args:
#         parser: Parsed arguments from argparse (should include arguments from
#               create_parser())

#     Returns:
#         tuple: (viewer, args) where viewer is configured based on args.viewer

#     Raises:
#         ValueError: If invalid viewer type or missing required arguments
#     """
#     import warp as wp  # noqa: PLC0415

#     import newton.viewer  # noqa: PLC0415

#     # parse args
#     if parser is None:
#         parser = create_parser()
#         args = parser.parse_known_args()[0]
#     else:
#         # When parser is provided, use parse_args() to properly handle --help
#         args = parser.parse_args()

#     # Set device if specified
#     if args.device:
#         wp.set_device(args.device)

#     # Create viewer based on type
#     if args.viewer == "gl":
#         viewer = newton.viewer.ViewerGL(headless=args.headless)
#     elif args.viewer == "usd":
#         if args.output_path is None:
#             raise ValueError("--output-path is required when using usd viewer")
#         viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
#     elif args.viewer == "rerun":
#         viewer = newton.viewer.ViewerRerun()
#     elif args.viewer == "null":
#         viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
#     else:
#         raise ValueError(f"Invalid viewer: {args.viewer}")

#     return viewer, args

# def run(example, args):
#     if hasattr(example, "gui") and hasattr(example.viewer, "register_ui_callback"):
#         example.viewer.register_ui_callback(lambda ui: example.gui(ui), position="side")

#     while example.viewer.is_running():
#         if not example.viewer.is_paused():
#             with wp.ScopedTimer("step", active=False):
#                 example.step()

#         with wp.ScopedTimer("render", active=False):
#             example.render()

#     if args is not None and args.test:
#         if not hasattr(example, "test"):
#             raise NotImplementedError("Example does not have a test method")
#         example.test()

#     example.viewer.close()
# # Namespace(device=None, viewer='gl', output_path='output.usd', num_frames=100, headless=False, test=False)
