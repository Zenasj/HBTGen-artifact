# torch.rand(1, dtype=torch.float64, device='cuda:0')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a = torch.tensor([[0]], device='cuda:0', dtype=torch.float64)
        a1 = a
        a2 = a1
        if x[0] >= 0:
            a.transpose(0, 1)  # This line is required for reproduction
            a2[0, 0] = 0 
        return (a1, )  # Returning a tuple as in original

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([1], device='cuda:0', dtype=torch.float64)

