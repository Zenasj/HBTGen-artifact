import torch

from timm import create_model

model = create_model('efficientnet_b3a', pretrained=True, scriptable=True)
torch.onnx._export(model, ...)