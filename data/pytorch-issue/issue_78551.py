import torch
print(torch.__version__)
yolov5s_cpu = torch.load('yolov5s.pt', 'cpu')
yolov5s_mps = torch.load('yolov5s.pt', 'mps')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')