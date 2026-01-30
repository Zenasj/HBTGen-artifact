import torch.nn as nn

import torch
import torchvision


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torchvision.transforms.functional.resize(x, size=[1024, 1024])
        return y


model = Model()
x = torch.rand(1, 3, 400, 500)
y = model(x)

onnx_model = torch.onnx.export(model, x, dynamo=True)

import torch
import torchvision
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = F.interpolate(x, size=[1024, 1024], mode='bilinear', align_corners=False)
        return y

model = Model()
x = torch.rand(1, 3, 400, 500)
y = model(x)

onnx_model = torch.onnx.export(model, x, dynamo=True)