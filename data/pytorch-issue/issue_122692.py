# torch.rand(5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, input_id):
        if input_id.is_complex():
            return torch.view_as_real(input_id)
        else:
            return input_id

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5)

