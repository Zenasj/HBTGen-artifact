# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.net1(x))
        return self.net2(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(20, 10)

