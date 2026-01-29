# torch.rand(1, 2, 3, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 4)  # 2*3*1 input features → 4 outputs
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 4D input to 2D
        x = self.linear(x)
        return self.relu(x)

def my_model_function():
    model = MyModel()
    # Initialize weights with Xavier uniform for numerical stability
    torch.nn.init.xavier_uniform_(model.linear.weight)
    return model

def GetInput():
    return torch.rand(1, 2, 3, 1, dtype=torch.float32)

