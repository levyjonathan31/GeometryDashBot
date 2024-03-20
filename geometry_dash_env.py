from screen_capture import capture_screen
import numpy as np
import keyboard
from gym import spaces
class gd_env:
    def __init__(self):
        self.action_space = spaces.Discrete(2)  # Discrete action space with 2 actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 152), dtype=np.uint8)
        self.state = None
        self.done = False
        self.thresh = None
    
    def step(self, action, progress):
        self.state, thresh = capture_screen()
        if np.array_equal(thresh, self.thresh) and self.done == False:
            print("Death frame: ", progress)
            return self.state, -1, True
        else:
            if action == 1:
                keyboard.press_and_release('space')
            self.thresh = thresh
            return self.state, 1, False

    def reset(self):
        self.state, thresh = capture_screen()
        self.done = np.array_equal(thresh, self.thresh)
        self.thresh = thresh
        return self.state
    
