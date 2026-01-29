# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 2)
        self.fc2 = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc1(x), self.fc2(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, dtype=torch.float32).cuda()

