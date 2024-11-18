import ultralytics as ultra
import torch.nn as nn
import torch

# YOLO --> Basic yolo architecture
# YOLO_WORLD --> Advanced yolo architecture. Requires more computational power
model = ultra.YOLO(model="yolo8n.pt", task = "detect") # task = "detect", no overriding in predict needed

infer = model.predict()
