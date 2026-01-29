# torch.rand(1, dtype=torch.int64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, t):
        # Extract scalar value from tensor input
        t_val = t.item()
        indices = torch.arange(0, t_val, dtype=torch.int64)
        return torch.nn.functional.one_hot(indices)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random integer between 1 and 20 as the input tensor
    return torch.randint(1, 21, (1,), dtype=torch.int64)

