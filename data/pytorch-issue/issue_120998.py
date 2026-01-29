# torch.rand(4096, 2048, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.view(torch.uint8) + 1

def my_model_function():
    return MyModel()

def GetInput():
    # Create a uint8 tensor and view as float16 to match input requirements
    data = torch.randint(0, 16, (4096, 4096), dtype=torch.uint8, device='cuda')
    return data.view(torch.float16)

