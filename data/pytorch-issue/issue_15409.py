# torch.rand(2, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  # Matches the Softmax usage in original issue
    
    def forward(self, x):
        return self.softmax(x)

def my_model_function():
    return MyModel()  # Directly returns the model instance

def GetInput():
    # Generate 2D tensor matching expected input for Softmax
    return torch.rand(2, 3, dtype=torch.float32)

