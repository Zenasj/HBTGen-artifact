# torch.rand(B, 10, dtype=torch.bfloat16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.fc(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 10, dtype=torch.bfloat16)

