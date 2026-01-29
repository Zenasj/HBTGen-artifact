# torch.rand(128, 1, 1, 1, dtype=torch.float64, device='cuda:0', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.ln = nn.LayerNorm(1)  # Normalize last dimension (size 1)

    def forward(self, x):
        return self.ln(x)

def my_model_function():
    model = MyModel()
    model.to(device='cuda:0', dtype=torch.float64)
    return model

def GetInput():
    return torch.rand(128, 1, 1, 1, device='cuda:0', dtype=torch.float64, requires_grad=True)

