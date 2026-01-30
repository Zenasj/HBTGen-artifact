import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.m = nn.AdaptiveMaxPool3d((1, None, None))
    def forward(self, x):
        x = self.m(x)
        return x

torch_model = TestModel()
dummy_input = torch.randn(1, 64, 10, 9, 8)

torch_out = torch.onnx.export(torch_model, dummy_input, 'test_model.onnx', verbose=True, opset_version=9)