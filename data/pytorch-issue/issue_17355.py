# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(10, 5)  # Example input shape: (B, 10)

    def forward(self, x):
        return F.log_softmax(self.layer(x), dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, dtype=torch.float32)

