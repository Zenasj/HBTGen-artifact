# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        h = x.size(2)
        w = x.size(3)
        scale_h = h / 100
        scale_w = w / 200
        return nn.functional.interpolate(
            x,
            scale_factor=(float(scale_h), float(scale_w)),
            mode="bicubic",
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

