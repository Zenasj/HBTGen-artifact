# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 10)  # Input size after cat
        
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = torch.cat([x1, x], dim=1)  # Uses torch.cat
        x3 = self.fc2(x2)
        values, indices = torch.topk(x3, 3)  # Uses torch.topk
        return values, indices

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Batch size (arbitrary choice)
    return torch.rand(B, 10, dtype=torch.float32)

