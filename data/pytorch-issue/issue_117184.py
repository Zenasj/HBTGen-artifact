# torch.rand(1, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.rand(2, 3, dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self, x):
        self.param.grad = torch.rand_like(self.param)
        self.param.grad = self.param.grad.to_sparse()  # Dynamo error occurs here
        return x  # Dummy return to satisfy forward contract

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32, device='cuda')

