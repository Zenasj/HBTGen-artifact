import torchvision

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
device = torch.device("cuda")
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

model.eval()
model.to(device)
model = torch.compile(model)
model(torch.randn(1,3,64,64).to(device))