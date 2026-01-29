# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 1)

    def forward(self, x):
        return self.lin(x)

def my_model_function():
    return MyModel()

def GetInput():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.ones(1, 10, requires_grad=True, device=device)

