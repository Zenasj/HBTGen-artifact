# torch.rand(1, 1, 1, 5, dtype=torch.float32)  # Inferred input shape from the example
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return F.logsigmoid(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 5, dtype=torch.float32)

