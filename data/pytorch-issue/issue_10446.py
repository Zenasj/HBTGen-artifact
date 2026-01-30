import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import shape_inference, utils

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, (100,100), mode='bilinear', align_corners=False)
        # x = F.relu(x)
        x = self.conv1(x)
        return x

filename = 'basic.onnx'
model = SimpleNet()
inputs = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, inputs, filename)

graph = onnx.load(filename)
graph = shape_inference.infer_shapes(graph)
print(graph.graph)

def forward(self, x):
        # x = F.interpolate(x, (100,100), mode='bilinear', align_corners=False)
        x = F.relu(x)
        x = self.conv1(x)
        return x