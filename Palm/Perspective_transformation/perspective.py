import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('project.avi')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
 
    #LowerLeft = 0,382
    #LowerRight = 640,382
    #UpperLeft = 120,283
    #UpperRight = 520,283
    origin_points = [[120,283],[520,283],[0,382],[640,382]]
    target_points = [[0,0],[350,0],[0,500],[350,500]]

    M1 = cv2.getPerspectiveTransform(np.float32(origin_points),np.float32(target_points))
    print(M1)

    res = cv2.warpPerspective(frame,M1,(350,500))

    cv2.imshow('Frame',res)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break
cap.release()
 
cv2.destroyAllWindows()



