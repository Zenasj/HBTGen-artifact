# torch.rand(B, 1, 10, dtype=torch.float32)  # Inferred input shape from the example
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, 10))  # 2D weight for ONNX compatibility (out_features=1, in_features=10)

    def forward(self, x):
        return F.linear(x, self.weight)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 10, dtype=torch.float32)

