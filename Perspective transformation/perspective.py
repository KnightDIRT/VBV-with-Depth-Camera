import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('project01.avi')

if not cap.isOpened():
    print("Error opening video stream or file")

origin_points = np.float32([[40, 300], [500, 300], [21, 440], [620, 440]])
target_points = np.float32([[0, 0], [350, 0], [0, 400], [350, 400]])
M1 = cv2.getPerspectiveTransform(origin_points, target_points)

# Initialize variables for frame smoothing
alpha = 1.0  # Adjust the value based on the desired smoothness
smoothed_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        res = cv2.warpPerspective(frame, M1, (350, 500))
        #set range of Red
        lowerHSV = [150,50,50]
        upperHSV = [180,255,255]
        
        lowerHSV1 = [140, 25, 75] 
        upperHSV1 = [150, 255, 255] 
        
        lowerHSV2 = [0, 25, 75] 
        upperHSV2 = [5, 255, 255] 
        
        cols,rows,ch = res.shape
        #Convert BGR to HSV
        hsv = cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,np.array(lowerHSV),np.array(upperHSV))
        mask1 = cv2.inRange(hsv,np.array(lowerHSV1),np.array(upperHSV1))
        mask2 = cv2.inRange(hsv,np.array(lowerHSV2),np.array(upperHSV2))
        maskall = mask + mask1 + mask2
        smoothed_mask = cv2.medianBlur(maskall, 5)
        cv2.imshow('Stabilized Frame', smoothed_mask)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()