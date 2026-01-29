# Input: two tensors of shape () with dtypes float32 and float64 respectively
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        add_res = torch.add(x, y)
        atan2_res = torch.atan2(x, y)
        dtypes_match = (add_res.dtype == atan2_res.dtype)
        return torch.tensor(dtypes_match, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn((), dtype=torch.float32)
    y = torch.randn((), dtype=torch.float64)
    return (x, y)

