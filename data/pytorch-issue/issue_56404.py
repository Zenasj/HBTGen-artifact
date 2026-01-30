import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.chunk(4, dim=1)

model = TestModel().eval()

x = torch.randn(1, 4, 256, 256)
torch.onnx.export(
    model,
    x,
    'test.onnx',
    export_params=True,
    opset_version=9) # change here between 9 and 11.