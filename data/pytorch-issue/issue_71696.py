# torch.rand(B, 0, 3, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=4)
    
    def forward(self, x):
        return self.gru(x)

def my_model_function():
    model = MyModel()
    model.to('cuda')
    return model

def GetInput():
    B = torch.randint(1, 10, (1,)).item()
    return torch.rand(B, 0, 3, dtype=torch.float32, device='cuda')

