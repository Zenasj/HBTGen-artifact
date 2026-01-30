import matplotlib.pyplot as plt
import torch

print(plt.get_backend())

# prints TkAgg

model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
print(plt.get_backend())

# prints agg

import torch
import matplotlib.pyplot as plt
import matplotlib

def get_yolo():
    b = plt.get_backend()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    matplotlib.use(b)
    return model