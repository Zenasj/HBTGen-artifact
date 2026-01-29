# torch.rand(2048, 2048, dtype=torch.float16), torch.rand(2048, 2048, dtype=torch.float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        A, B = x
        return A @ B

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.randn(2048, 2048, dtype=torch.float16).cuda()
    B = torch.randn(2048, 2048, dtype=torch.float16).cuda()
    return (A, B)

