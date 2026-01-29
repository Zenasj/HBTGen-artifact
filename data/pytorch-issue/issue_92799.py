# Input: (torch.randint(-16,4, (), dtype=torch.int64), torch.randint(0,1, (1,), dtype=torch.uint8))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        arg1, arg2 = inputs
        return torch.eq(arg1, arg2)

def my_model_function():
    return MyModel()

def GetInput():
    arg1 = torch.randint(-16, 4, size=(), dtype=torch.int64)
    arg2 = torch.randint(0, 1, size=(1,), dtype=torch.uint8)
    return (arg1, arg2)

