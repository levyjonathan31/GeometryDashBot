import time
import cv2
import dxcam
import matplotlib.pyplot as plt
camera = dxcam.create()
def capture_screen():
    # Create a camera object
    # Get the screen info
    # start = time.time()
    # Grab the screen
    state = camera.grab()
    while(state is None):
        state = camera.grab()
    pos_state = state[200:230, 40:140]   
    state = cv2.resize(state, (256, 128))
    state = state[:, 104:, :]
    cv2.imwrite("screenshot.png", state)
    state = state.reshape(3, 128, 152)
    # Convert to PIL image
    gray = cv2.cvtColor(pos_state, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imwrite("screenshots/pos_state" + str(time.time()) + ".png", thresh)
    # print("Time to process screen: ", time.time() - start)
    return state, thresh

# capture_screen()