# torch.rand(2, 3, 4, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        ret = x.max(dim=0)
        # This line triggers the structseq vs tuple issue described in the PR
        return torch.max(x, dim=0, out=ret)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4, 5, dtype=torch.float32)

