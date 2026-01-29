# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class HasGraphBreak(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 5)

    def forward(self, x):
        x = self.linear1(x)
        torch._dynamo.graph_break()
        return self.linear2(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule = HasGraphBreak()

    def forward(self, x):
        x = torch.relu(x)
        x = self.submodule(x)
        return torch.relu(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 5)

