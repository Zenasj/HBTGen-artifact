import torch
import torch.nn as nn

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.flatten()


x = torch.rand(100)
torch.onnx.export(Flatten(), x, 'flatten.onnx', opset_version=13)

print(Flatten()(x).shape)
print(onnx.load("flatten.onnx").graph.output[0])