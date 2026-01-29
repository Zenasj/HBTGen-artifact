# torch.rand(4, 155, 4096, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters matching the original repro's shapes and requires_grad settings
        self.weight = nn.Parameter(torch.randn(12288, 4096))  # weight from F.linear
        self.y = nn.Parameter(torch.randn(4, 155, 12288, requires_grad=True))  # y from f()

    def forward(self, x):
        # Emulate g(x) = x + 1 (from the original compiled function)
        x = x + 1
        # Emulate F.linear(x, weight) + y (from f())
        result = F.linear(x, self.weight, bias=None)
        # In-place addition which triggers the view/in-place modification error
        result += self.y
        return result

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the original input shape and requires_grad=True
    return torch.randn(4, 155, 4096, requires_grad=True)

