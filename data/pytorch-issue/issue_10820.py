# torch.rand(2, 3, dtype=torch.float32)  # 2D input tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        x[0, 0] = 100  # In-place modification causing ONNX export issue
        y = torch.cat([x, x], dim=0)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float32)

