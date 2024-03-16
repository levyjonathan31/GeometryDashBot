import time
import cv2
import mss
import numpy as np
from PIL import Image
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
def capture_screen():
    # Get the screen info
    with mss.mss() as sct:  
    # Part of the screen to capture
        # Get raw pixels from the screen, save it to a Numpy array
        state = np.array(sct.grab(monitor))
        # Convert to rgb 
        pos_state = state[200:230, 40:140, :3]   
        state = state[:, :, :3]
        state = cv2.resize(state, (128, 64))
        state = state[:, 40:, :]
        state = state.reshape(3, 64, 88)
        # Convert to PIL image
        gray = cv2.cvtColor(pos_state, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # cv2.imwrite("screenshots/pos_state" + str(time.time()) + ".png", thresh)

        return state, thresh

capture_screen()