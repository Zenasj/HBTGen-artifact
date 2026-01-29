# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        length = x.shape[-1]
        if length <= 1:
            return x
        x_0 = x[..., :length//2]
        x_1 = x[..., length//2:]
        y_1 = self.forward(x_1)  # Recursive call via self
        y_0 = x_0
        y = torch.cat([y_0, y_1], dim=-1)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float32)

