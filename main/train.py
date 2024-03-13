from ultralytics import YOLO
import torch

torch.cuda.set_device(0) # Set to your desired GPU number
 
# Load the model.
model = YOLO("yolov8n.pt") 
# Training.
results = model.train(
   data= "dataset/CV Project.v2i.yolov8/data.yaml",
   imgsz=640,
   epochs=10,
   batch=12,
   name='test1',
   workers=0,
   device=0
)
