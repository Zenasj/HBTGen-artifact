# torch.rand(B, 16, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size (can be adjusted)
    return torch.rand(B, 16, dtype=torch.float32)

