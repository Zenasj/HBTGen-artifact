# torch.rand(2, 3, 4), torch.rand(2, 3, 4)  # two tensors of same shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.original = OriginalModel()
        self.workaround = WorkaroundModel()

    def forward(self, inputs):
        b, c = inputs
        orig_out = self.original(b, c)
        work_out = self.workaround(b, c)
        return torch.all(orig_out == work_out)  # Returns True if outputs match

class OriginalModel(nn.Module):
    def forward(self, b, c):
        a = b > c
        return ~a  # Uses bitwise_not (problematic in ONNX)

class WorkaroundModel(nn.Module):
    def forward(self, b, c):
        a = b > c
        return a == torch.as_tensor(False)  # ONNX-compatible workaround

def my_model_function():
    return MyModel()

def GetInput():
    b = torch.rand(2, 3, 4)
    c = torch.rand(2, 3, 4)
    return (b, c)  # Tuple of two tensors with same shape

