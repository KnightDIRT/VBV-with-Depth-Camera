from ultralytics import YOLO
import cv2
import math 
import pyrealsense2 as rs
import numpy as np
import torch 

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

model = YOLO("best.pt")

classNames = ["Go","Pole","Stop","Yield"]


def perspective_transform(img):

        origin_points = np.float32([[140, 450], [1140, 450], [50, 700], [1230, 700]])
        target_points = np.float32([[0, 0], [350, 0], [0, 400], [350, 400]])
        M1 = cv2.getPerspectiveTransform(origin_points, target_points)
        alpha = 1.0 
        smoothed_frame = None
        
        res = cv2.warpPerspective(img, M1, (350, 500))
      
        lowerHSV = [150,25,75]
        upperHSV = [200,255,255]
        
        lowerHSV1 = [140, 25, 75] 
        upperHSV1 = [150, 255, 255] 
        
        lowerHSV2 = [6, 25, 75] 
        upperHSV2 = [10, 255, 255] 
        
        cols,rows,ch = res.shape
        
        hsv = cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,np.array(lowerHSV),np.array(upperHSV))
        mask1 = cv2.inRange(hsv,np.array(lowerHSV1),np.array(upperHSV1))
        mask2 = cv2.inRange(hsv,np.array(lowerHSV2),np.array(upperHSV2))
        maskall = mask + mask1 + mask2
        smooth_mask = cv2.medianBlur(maskall, 5)
        return smooth_mask
        
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def merge_similar_lines(lines, delta_slope=0.1, delta_intercept=50):
    if lines is None:
        return None
    
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for line in lines for x1, y1, x2, y2 in line]
    intercepts = [y1 - slope * x1 for slope, line in zip(slopes, lines) for x1, y1, x2, y2 in line]
    clusters = []

    for idx, (line, slope, intercept) in enumerate(zip(lines, slopes, intercepts)):
        merged = False
        for cluster in clusters:
            if abs(cluster['slope'] - slope) < delta_slope and abs(cluster['intercept'] - intercept) < delta_intercept:
                cluster['lines'].append(line[0])
                cluster['slope'] = np.mean([slope, cluster['slope']])
                cluster['intercept'] = np.mean([intercept, cluster['intercept']])
                merged = True
                break
        if not merged:
            clusters.append({'lines': [line[0]], 'slope': slope, 'intercept': intercept})
    
    # Averaging lines in each cluster
    merged_lines = []
    for cluster in clusters:
        xs = []
        ys = []
        for line in cluster['lines']:
            xs.append(line[0])
            xs.append(line[2])
            ys.append(line[1])
            ys.append(line[3])
        
        avg_x1 = min(xs)
        avg_y1 = int(cluster['slope'] * avg_x1 + cluster['intercept'])
        avg_x2 = max(xs)
        avg_y2 = int(cluster['slope'] * avg_x2 + cluster['intercept'])
        
        merged_lines.append([(avg_x1, avg_y1, avg_x2, avg_y2)])
    
    return merged_lines

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    merged_lines = merge_similar_lines(lines)

    if merged_lines is not None:
        for line in merged_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 100, 255), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def anglelinesfil(lines, min_angle=30, max_angle=150):
    if lines is None:
        return None
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if min_angle <= angle <= max_angle:
                filtered_lines.append(line)
    return filtered_lines

def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
   
    region_of_interest_vertices = [
        (0, height),  # Bottom left
        (width, height),  # Bottom right
        (width, height*0.8),  # Middle right
        (width*0.5, height*0.5),  # Middle center 
        (0, height*0.8),  # Middle left 
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    #cv2.imshow('Gray Image', gray_image)
    
    canny_image = cv2.Canny(gray_image, 80, 100)
    
 
    #cv2.imshow('Canny Image', canny_image)
    
    cropped_image = region_of_interest(
        canny_image,
        np.array([region_of_interest_vertices], np.int32),
    )
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), (255,) * image.shape[2])
    masked_image = cv2.bitwise_and(image, mask)
    cv2.imshow('Cropped Image', masked_image)
    
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=10,
        maxLineGap=200
    )
    
    filtered_lines = anglelinesfil(lines)
    image_with_lines = draw_the_lines(image, filtered_lines)
    
    return image_with_lines


while True:

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
  
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = np.asanyarray(color_frame.get_data())

    lane_results = perspective_transform(rgb_image)
    cv2.imshow("top-view",lane_results)
    
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    results = model(rgb_image, stream=True, conf=0.80)
    
    #lane detection
    rgb_image = process(rgb_image)

    for r in results:
        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

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

