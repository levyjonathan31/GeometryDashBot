from screen_capture import capture_screen
import numpy as np
import keyboard
class env:
    def __init__(self):
        self.action_space = [0, 1]
        self.state = None
        self.done = False
        self.thresh = None
    
    def step(self, action, prev_reward):
        self.state, self.done, thresh = capture_screen()
        if np.array_equal(thresh, self.thresh):
            self.done = True
            print("Game Over")
        elif action == 1:
            keyboard.press_and_release('space')
        self.thresh = thresh
        return self.state, prev_reward + 0.01*(1 - int(self.done)), self.done, None
    def reset(self):
        self.state, self.done, thresh = capture_screen()
        if np.array_equal(thresh, self.thresh):
            self.done = True
        self.thresh = thresh
        return self.state