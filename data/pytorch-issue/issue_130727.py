# Input shapes: (1920, 1, 100), (1920, 256, 512), (1920, 512, 100), dtype=torch.bfloat16
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        input_tensor, batch1, batch2 = inputs
        return torch.baddbmm(input_tensor, batch1, batch2)

def my_model_function():
    return MyModel()

def GetInput():
    device = torch.device("cuda")
    input_tensor = torch.rand(1920, 1, 100, device=device, dtype=torch.bfloat16)
    input_tensor = torch.as_strided(input_tensor, (1920, 1, 100), (100, 100, 1))
    batch1 = torch.rand(1920, 256, 512, device=device, dtype=torch.bfloat16)
    batch1 = torch.as_strided(batch1, (1920, 256, 512), (512, 983040, 1))
    batch2 = torch.rand(1920, 512, 100, device=device, dtype=torch.bfloat16)
    batch2 = torch.as_strided(batch2, (1920, 512, 100), (51200, 100, 1))
    return (input_tensor, batch1, batch2)

