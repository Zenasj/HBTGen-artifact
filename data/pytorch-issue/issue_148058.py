# torch.rand(1, 2, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        x = F.gumbel_softmax(x, tau=1.0, hard=True)
        x = torch.where(x > 0.5, x, torch.zeros_like(x))
        # Fixed index as in original repro
        index = torch.ones(1, 2, dtype=torch.long, device=x.device)
        x = torch.scatter(x, dim=1, index=index, src=torch.ones_like(x))
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 2, dtype=torch.float32)

