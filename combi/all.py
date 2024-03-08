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

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 100, 255), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

while True:

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
  
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = np.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    results = model(rgb_image, stream=True, conf=0.80)

    #lane detection
    rgb_image = process(rgb_image)

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

