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
            return self.state, -1, self.done, None
        elif action == 1:
            keyboard.press_and_release('space')
        self.thresh = thresh
        # print("Prev Reward: ", prev_reward, "New Reward: ", 0.1 + 0.001*prev_reward)
        return self.state, 0.1 + 0.001*prev_reward, self.done, None
    def reset(self):
        self.state, self.done, thresh = capture_screen()
        if np.array_equal(thresh, self.thresh):
            self.done = True
        self.thresh = thresh
        return self.state