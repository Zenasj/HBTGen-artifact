# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # Example layer; inferred from common practice

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 10)  # Batch size 5, input features 10

