import torch
from torch import nn, jit
from typing import List

# torch.rand(B, 1, 5, 5, 5, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)

    def forward(self, x):
        outputs = jit.annotate(List[torch.Tensor], [])
        for i in range(x.size(0)):
            outputs.append(self.conv(x[i].unsqueeze(0)))
        return torch.stack(outputs, 0).squeeze()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((3, 1, 5, 5, 5), dtype=torch.float32)

