import polyscope as ps
import polyscope.imgui as psim

class UI:
    def __init__(self, update = lambda: None):
        self.time_step = 0
        self.time_step_size = 0.1
        self.run = True

        ps.init()

    def restart(self):
        self.time_step = 0
        ps.init()
    
    def set_callback(self, callback):
        ps.set_user = callback
    

    