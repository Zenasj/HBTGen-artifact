# torch.rand(1, 1, 1, 1, dtype=torch.float32)  # Inferred input shape based on example scalar usage
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        values = ()
        values += (x,)
        return values  # Returns a tuple containing the input tensor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1)  # 4D tensor to match required input shape format

