# torch.rand(B, 100, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        layers = []
        # 6 hidden layers of Linear + ReLU as per benchmark description
        for _ in range(6):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 10000 as per benchmark, but reduced to 1 for minimal example
    # Actual benchmark used 10000, but this is variable
    return torch.rand(1, 100, dtype=torch.float32)

