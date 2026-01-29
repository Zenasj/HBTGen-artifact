# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.to(torch.int32)

def my_model_function():
    return MyModel()

def GetInput():
    max_int32 = torch.iinfo(torch.int32).max
    # Creates a tensor with value exactly at the max_int32, which due to float precision
    # may overflow when cast to int32 (as per the issue's example)
    return torch.tensor([max_int32], dtype=torch.float32)

