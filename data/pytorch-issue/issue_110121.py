# torch.rand(3, 4, 5, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 7)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    model.eval()
    model.cuda()  # Matches device context in original issue's sample code
    return model

def GetInput():
    return torch.rand(3, 4, 5, device='cuda')  # Matches sample input dimensions and device

