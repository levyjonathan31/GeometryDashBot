from screen_capture import capture_screen
import numpy as np
import keyboard
import gym
from gym import spaces

class gd_env:
    def __init__(self):
        self.action_space = spaces.Discrete(2)  # Discrete action space with 2 actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 108, 192), dtype=np.uint8)
        self.state = None
        self.done = False
        self.thresh = None
    
    def step(self, action):
        self.state, thresh = capture_screen()

        if action == 1:
            keyboard.press_and_release('space')

        if np.array_equal(thresh, self.thresh):
            print("Game Over")
            return self.state, -0.5, True
        else:
            self.thresh = thresh
            return self.state, 0.1, False

    def reset(self):
        self.state, thresh = capture_screen()
        self.done = np.array_equal(thresh, self.thresh)
        self.thresh = thresh
        return self.state
