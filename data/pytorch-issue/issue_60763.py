import torch
import torch.nn as nn
import torch.nn.functional as F

# Inputs: (x (5, 20), y (5, 20), w (40, 20), b (40)) (all dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, w, b = inputs
        z = torch.cat([x, y], dim=0)
        li = F.linear(z, w, b)
        return F.layer_norm(li, li.size())

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(5, 20)
    y = torch.rand(5, 20)
    w = torch.rand(40, 20)
    b = torch.rand(40)
    return (x, y, w, b)

