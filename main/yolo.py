from ultralytics import YOLO
import cv2
import math 
import pyrealsense2 as rs
import numpy as np
import torch 

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

model = YOLO("best.pt")

classNames = ["Go","Pole","Stop","Yield"]

while True:

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
  
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = np.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    results = model(rgb_image, stream=True, conf=0.75)

    for r in results:
        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            center_x = int((x1 + x2)/2)
            center_y = int((y1 + y2)/2)
            #print(str(center_x) + "," + str(center_y))
            dist = depth_frame.get_distance(center_x, center_y)
            if dist != 0:
                real_dist = f"{dist:.3f}"
            else:
                real_dist = "nan"

            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            val = str(classNames[cls]) + "," + "Dist:" + str(real_dist)

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 255, 0)
            thickness = 2

            text_size = cv2.getTextSize(val, font, fontScale, thickness)[0]
            cv2.rectangle(rgb_image, (x1, y1), (x1 + text_size[0], y1 - text_size[1]), (0, 0, 0), -1)
            cv2.putText(rgb_image, val, org, font, fontScale, (255,255,255), thickness)
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow('Webcam', rgb_image)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()

