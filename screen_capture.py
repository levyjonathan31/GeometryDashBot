import time
import cv2
import mss
import numpy
from config import monitor
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
def capture_screen():
    # Get the screen info
    with mss.mss() as sct:  
    # Part of the screen to capture
        # Get raw pixels from the screen, save it to a Numpy array
        state = numpy.array(sct.grab(monitor))
        pos_state = state[200:225, 45:200, :3]   
        state = cv2.resize(state, (192, 108))
        # Convert to PIL image
        gray = cv2.cvtColor(pos_state, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # cv2.imwrite("pos_state.png", thresh)
        # cv2.imwrite("screenshot.png", state)
        return state, False, thresh