import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x, y):
        assert x.dtype == y.dtype
        return torch.maximum(x, y)

torch.onnx.export(Model(), (torch.randn(1, 2), torch.randn(1, 2)), "model.onnx",
    dynamic_axes={"0": {0: "batch", 1: "width"}, "1": {0: "batch", 1: "width"}, "2": {0: "batch", 1: "width"}})

class Model2(nn.Module):
    def forward(self, x, y):
        assert x.dtype == y.dtype
        return torch.where((x > y) | (torch.isnan(x)), x, y)