import torch.nn as nn
import torchvision

import io
import onnx
import torch
from torchvision.models import resnet18

buffer = io.BytesIO()
torch.onnx.export(resnet18(), torch.randn(1, 3, 224, 224), buffer)
buffer.seek(0, 0)
onnx_model = onnx.load(buffer)
for node in onnx_model.graph.node:
    print(node.name)

# Run this without patching numerical atom `_unqualified_variable_name`
class MainModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = torch.nn.ModuleList(
            [torch.nn.Linear(10, 10) for _ in range(2)]
        )

    def forward(self, x):
        y = self.module_list[0](x)
        z = self.module_list[1](y)
        return z

module = MainModule()
print(module)
f = io.BytesIO()
torch.onnx.export(module, torch.randn(1, 10), f, verbose=True)
f.seek(0, 0)
onnx_model = onnx.load(f)
for node in onnx_model.graph.node:
    print(node.name)

module = torch.hub.load("intel-isl/MiDaS:f28885af", "MiDaS_small")
input = torch.randn(1, 3, 320, 640)