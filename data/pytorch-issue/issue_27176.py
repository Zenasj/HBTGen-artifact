import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

print(torch.__version__)
print(onnx.__version__)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x


torch_model = TestModel()
dummy_input = torch.randn(1, 3, 256, 256)

torch_out = torch.onnx.export(torch_model, dummy_input, 'model.onnx', verbose=True, opset_version=11)

onnx_model = onnx.load('model.onnx')
print(onnx_model)
onnx.checker.check_model(onnx_model)