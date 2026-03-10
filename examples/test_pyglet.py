import warp as wp
import numpy as np
import pyglet

pyglet.options["headless"] = True
pyglet.options["shadow_window"] = False

from pyglet.gl import *

wp.init()

@wp.kernel
def my_kernel(arr: wp.array(dtype=wp.float32)):
    i = wp.tid()
    arr[i] *= 2.0


# create a hidden window to get a current GL context
window = pyglet.window.Window(width=1, height=1, visible=False)
window.switch_to()

try:
    # create a GL buffer
    gl_buffer_id = GLuint()
    glGenBuffers(1, gl_buffer_id)

    # copy some data to the GL buffer
    glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_id)
    gl_data = np.arange(1024, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, gl_data.nbytes, gl_data.ctypes.data, GL_DYNAMIC_DRAW)
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
    window.close()