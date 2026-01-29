# torch.rand(3, 3, dtype=torch.float32, device='cuda')
import torch
import torchaudio  # Required to trigger symbol conflict

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.sin(x) + torch.cos(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32, device='cuda')

