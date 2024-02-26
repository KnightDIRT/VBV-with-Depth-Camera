import cv2
import numpy as np
import matplotlib.pyplot as plt

depth = cv2.imread("D:\\vision1\\Depth\\1314.png")
#depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

cv2.imshow("depth", depth)
cv2.waitKey(0)
cv2.destroyAllWindows()