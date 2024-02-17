import numpy as np

class FrameStacker:
    def __init__(self, initial_frame, stack_size=10):
        self.stack = np.stack([initial_frame]*stack_size)

    def add_frame(self, new_frame):
        self.stack = np.roll(self.stack, shift=-1)
        self.stack[-1] = new_frame

    def get_state(self):
        return self.stack
    