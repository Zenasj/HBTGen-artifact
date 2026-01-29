# torch.rand(B, 100, dtype=torch.float32)  # Assuming a batch size B and input features of size 100
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 50, bias=False)
        self.l2 = nn.Linear(50, 1, bias=False)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Assuming a batch size of 32 for demonstration purposes
    B = 32
    return torch.rand(B, 100, dtype=torch.float32)

