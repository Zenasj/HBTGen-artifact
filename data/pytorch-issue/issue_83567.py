# torch.rand(B, N), torch.tensor(5, dtype=torch.int64)  # Input tensors (x and y)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        x, y = inputs
        return x[0, y:]

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(2, 10)  # Example input: batch size 2, features 10
    y = torch.tensor(3)    # Scalar index tensor
    return (x, y)

