from screen_capture import capture_screen

import keyboard
class env:
    def __init__(self):
        self.action_space = [0, 1]
        self.state = None
        self.done = False
        self.text = ""
    
    def step(self, action):
        self.state, self.done, text = capture_screen()
        if (text == self.text):
            self.done = True
            print("Game Over")
        elif action == 1:
            keyboard.press_and_release('space')
        self.text = text
        return self.state, 0.04*(-0.2*action + 1 - int(self.done)), self.done, None
    def reset(self):
        self.state, self.done, text = capture_screen()
        self.done = False
        self.text = ""
        return self.state