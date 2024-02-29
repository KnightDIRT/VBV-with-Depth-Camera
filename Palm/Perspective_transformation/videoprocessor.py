import cv2
import numpy as np
from imutils import paths
import re

def sort_key_func(file_path):
    """Extracts numbers from the filename and returns them to ensure proper sorting."""
    numbers = re.findall(r'\d+', file_path)
    return [int(num) for num in numbers]

img_array = []

# Use the actual path where your images are stored
image_folder_path = r'C:\Users\papho\OneDrive\Desktop\vision1\RGB'

image_paths = list(paths.list_images(image_folder_path))
# Sort the image paths using the custom sort function
image_paths = sorted(image_paths, key=sort_key_func)

size = None  # To be determined based on the first image

for image_path in image_paths:
    image = cv2.imread(image_path)
    if size is None:
        height, width, layers = image.shape
        size = (width, height)
    img_array.append(image)

# Ensure you are using a codec compatible with the file extension
out = cv2.VideoWriter('project02.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for img in img_array:
    out.write(img)
out.release()