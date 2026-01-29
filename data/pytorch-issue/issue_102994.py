# torch.rand(B, 4, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 2)

    def forward(self, input):
        out = F.relu(self.l1(input))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, dtype=torch.float32)

