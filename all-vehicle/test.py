import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', 'resources/weights/last_digit_256_05052023_au.pt')
model.conf = 0.4
model.imgsz = 256

image = cv2.imread("1.jpg")
result = model([image], agnostic_nms=True, conf_thres=0.4, size=256)
print(result.pandas().xyxy[0])