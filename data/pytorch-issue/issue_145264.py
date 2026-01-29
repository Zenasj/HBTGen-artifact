# torch.rand(8, 1, 4, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(1, 25, bias=True)  # Matches original model's Linear parameters

    def forward(self, x):
        # Replicate operations from the original function
        interpolated = F.interpolate(
            x,
            size=(36, 1),
            mode="bicubic",
            align_corners=None,
            antialias=False,
        )
        negated = torch.neg(interpolated)
        linear_out = self.linear_layer(negated)
        casted = linear_out.to(torch.float64)  # Explicit dtype conversion
        tan_out = torch.tan(casted)
        return tan_out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 1, 4, 1, dtype=torch.float32)

