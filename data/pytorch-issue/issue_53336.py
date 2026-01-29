# Two input tensors: torch.rand(2,3, device='cuda'), torch.rand(3,3, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        mat1, mat2 = inputs
        return torch.mm(mat1, mat2)

def my_model_function():
    return MyModel()

def GetInput():
    mat1 = torch.randn(2, 3).cuda()
    mat2 = torch.randn(3, 3).cuda()
    return (mat1, mat2)

