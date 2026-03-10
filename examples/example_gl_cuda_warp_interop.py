import warp as wp
import numpy as np
import glfw
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    glBindBuffer,
    glBufferData,
    glDeleteBuffers,
    glGenBuffers,
)

wp.init()


@wp.kernel
def my_kernel(arr: wp.array(dtype=wp.float32)):
    i = wp.tid()
    arr[i] *= 2.0


def create_headless_egl_context():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    # Request EGL so this works in SSH/headless setups.
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

    window = glfw.create_window(1, 1, "headless", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create EGL OpenGL context (check EGL/NVIDIA driver on node)")

    glfw.make_context_current(window)
    return window


window = create_headless_egl_context()

try:
    # create a GL buffer
    gl_buffer_id = glGenBuffers(1)

    # copy some data to the GL buffer
    glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_id)
    gl_data = np.arange(1024, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, gl_data.nbytes, gl_data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # register the GL buffer with CUDA
    cuda_gl_buffer = wp.RegisteredGLBuffer(gl_buffer_id)

    # map the GL buffer to a Warp array
    arr = cuda_gl_buffer.map(dtype=wp.float32, shape=(1024,))

    # launch a Warp kernel to manipulate or read the array
    wp.launch(my_kernel, dim=1024, inputs=[arr])

    # unmap the GL buffer
    cuda_gl_buffer.unmap()

finally:
    try:
        glDeleteBuffers(1, [gl_buffer_id])
    except Exception:
        pass
    glfw.destroy_window(window)
    glfw.terminate()