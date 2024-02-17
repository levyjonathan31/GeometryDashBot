from screeninfo import get_monitors
import cv2
screen = get_monitors()[0]
monitor = {"top": 0, "left": 0, "width": screen.width, "height": screen.height}
print(monitor)