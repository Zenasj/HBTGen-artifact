import torch
import torchvision
model = torchvision.models.resnet18()
torch.onnx.export(model, torch.randn(1,3,224,224), 'resnet18.onnx', input_names=['input'], output_names=['output'])

import torch
import torchvision
model = torchvision.models.resnet18(pretrained=True)
torch.onnx.export(model, torch.randn(1,3,224,224), 'resnet18.onnx', input_names=['input'], output_names=['output'])