# 8 tensors of shape torch.Size([100, 1000]) with dtype=torch.float32 packed into a tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x1, y1, w1, h1, x2, y2, w2, h2 = inputs
        xi = torch.max(x1, x2)
        yi = torch.max(y1, y2)
        wi = torch.clamp(torch.min(x1 + w1, x2 + w2) - xi, min=0.)
        hi = torch.clamp(torch.min(y1 + h1, y2 + h2) - yi, min=0.)
        area_i = wi * hi
        area_u = w1 * h1 + w2 * h2 - wi * hi
        return area_i / torch.clamp(area_u, min=1e-5)

def my_model_function():
    return MyModel()

def GetInput():
    return tuple(torch.rand(100, 1000, dtype=torch.float32) for _ in range(8))

