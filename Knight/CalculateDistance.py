import cv2
import numpy as np
import matplotlib.pyplot as plt

RootDir = r"C:\Users\papho\OneDrive\Desktop\vision5"

RGBDir = RootDir + "\\RGB"
DepthDir = RootDir + "\\Depth"

n = np.random.randint(0, 1979)
rgb = cv2.imread(RGBDir + f"\\{n}.png")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = cv2.imread(DepthDir + f"\\{n}.png")
depthHSV = cv2.cvtColor(depth, cv2.COLOR_BGR2HSV)
depthHSV[..., 1] = 255
depthHSV[..., 2] = 255
depth = cv2.cvtColor(depthHSV, cv2.COLOR_HSV2RGB)
hue_shift = 120
depthHSV[..., 0] = (-depthHSV[..., 0] + hue_shift) % 180
#depthHSV[..., 0] = 120
depthNew = cv2.cvtColor(depthHSV, cv2.COLOR_HSV2RGB)
print("IMG_NUM:",n)

def mouse_callback(event, x, y, flags, param):
    # Check for left mouse click event (event == cv2.EVENT_LBUTTONDOWN)
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked coordinates
        global clicked_coordinates
        clicked_coordinates = (x, y)
        print(f"COORD: ({x}, {y})")
        print("HUE:", depthHSV[y,x,0])
        print("DIST:", depthHSV[y,x,0] / 120 * 3)

# Initialize an empty list to store clicked coordinates
clicked_coordinates = []

# Create a named window for displaying the image
cv2.namedWindow("depth", cv2.WINDOW_NORMAL)

# Set the mouse callback function
cv2.setMouseCallback("depth", mouse_callback)

cv2.imshow("depth", depthNew)


cv2.waitKey(0)
cv2.destroyAllWindows()