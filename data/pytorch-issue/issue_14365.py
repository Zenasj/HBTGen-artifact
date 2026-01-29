# torch.randint(4, (B,), dtype=torch.long, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lut_dummy = nn.Embedding(1, 1, max_norm=1.0).to("cuda")
        self.lut_a = nn.Embedding(22, 256, max_norm=1.0).to("cuda")

    def forward(self, src):
        return self.lut_a(src)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(4, (2,), dtype=torch.long, device='cuda')

