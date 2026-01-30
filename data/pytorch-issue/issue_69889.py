import onnx
import torch
import logging
from tanet_arch.tanet import TANet


model = TANet('resnet18', 1, 1, 0, 4, True, True)
device = None if torch.cuda.is_available() else torch.device('cpu')
im = torch.zeros(1, 3, 640, 256).to(device)  # image size(1,3,640,256) BCHW iDetection

torch.onnx.export(model,
    (im, im),
    "./model.onnx",
    verbose=True,
    opset_version=17
)