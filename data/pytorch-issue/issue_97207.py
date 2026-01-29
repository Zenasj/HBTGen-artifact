# torch.rand(3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Trigger multiple TypedStorage deprecation warnings
        x.storage()  # First call
        x.storage()  # Second call
        x.storage()  # Third call
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

