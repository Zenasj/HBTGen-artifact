# torch.rand(2, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        ix = x + 1
        a = ix.transpose(0, 1)
        return a.detach(), a  # Returns detached view and original alias

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float32)

