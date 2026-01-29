# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    model = MyModel()
    model.to('cuda')  # Matches the reported environment setup
    return model

def GetInput():
    # Matches input shape and device from the issue's example (2 samples, 3 features)
    return torch.randn(2, 3, device='cuda', dtype=torch.float32)

