import torch.nn as nn

py
import torch

class Net(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(C, eps=1e-8)
    def forward(self, x):
        return self.layer_norm(x)

N, C = 8, 4
model = Net(C).cuda().half()
x = torch.randn(N, C).cuda().half()

torch.onnx.export(model, x, "test_layernorm_export_fp16.onnx", opset_version=12)