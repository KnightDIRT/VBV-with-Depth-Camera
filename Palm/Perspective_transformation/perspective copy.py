import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('project01.avi')

if not cap.isOpened():
    print("Error opening video stream or file")

origin_points = np.float32([[200, 210], [440, 210], [0, 240], [640, 240]])
target_points = np.float32([[0, 0], [350, 0], [0, 400], [350, 400]])
M1 = cv2.getPerspectiveTransform(origin_points, target_points)

# Initialize variables for frame smoothing
alpha = 0.2  # Adjust the value based on the desired smoothness
smoothed_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        res = cv2.warpPerspective(frame, M1, (350, 500))
        
        if smoothed_frame is None:
            smoothed_frame = res.copy()
        else:
            cv2.addWeighted(res, alpha, smoothed_frame, 1 - alpha, 0, smoothed_frame)

        cv2.imshow('Stabilized Frame', smoothed_frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
