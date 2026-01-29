# Inputs: two tensors of shape (3, 3) on CUDA device
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        z1 = torch.mm(x, y)
        z2 = self.gn(x, y)
        return z1 + z2

    def gn(self, x, y):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = torch.mm(x, y)
            torch._dynamo.graph_break()
            return torch.sin(z)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(3, 3, device="cuda"), torch.rand(3, 3, device="cuda"))

