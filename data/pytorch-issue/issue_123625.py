# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 5)
        
    def forward(self, x):
        return self.linear(x)
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass  # Override to return empty state_dict

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Arbitrary batch size
    return torch.rand(B, 3, dtype=torch.float32)

