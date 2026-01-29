# torch.rand(4, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, device=device), nn.Linear(4, 4, device=device)])
        self.output = nn.Linear(4, 4, device=device)

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)
        return self.output(z)

def my_model_function():
    device = torch.device("cuda")
    return MyModel(device)

def GetInput():
    return torch.randn(4, 4, device="cuda")

