import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('project02.avi')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
 
    #LowerLeft = 0,382
    #LowerRight = 640,382
    #UpperLeft = 120,283
    #UpperRight = 520,283
    origin_points = [[200,210],[440,210],[0,240],[640,240]]
    target_points = [[0,0],[350,0],[0,400],[350,400]]

    M1 = cv2.getPerspectiveTransform(np.float32(origin_points),np.float32(target_points))
    print(M1)

    res = cv2.warpPerspective(frame,M1,(350,500))
    # hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # lower_y = np.array([18, 94, 140])
    # upper_y = np.array([18, 255, 255])

    # mask = cv2.inRange(hsv, lower_y, upper_y)
    # edges = cv2.Canny(mask, 74, 150)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(or_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.imshow('Frame1',res)

 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break
cap.release()
 
cv2.destroyAllWindows()



