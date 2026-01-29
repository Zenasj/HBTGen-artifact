# Input is a tuple of two tensors with shapes (B, M, 0) and (B, 0, N), e.g., torch.rand(3,2,0).cuda(0), torch.rand(3,0,3).cuda(0)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        t1, t2 = inputs
        bmm_result = torch.bmm(t1, t2)
        matmul_result = torch.matmul(t1, t2)
        return bmm_result, matmul_result  # Return both results for comparison

def my_model_function():
    return MyModel()

def GetInput():
    B, M, N = 3, 2, 3  # Dimensions from the issue's test case
    t1 = torch.rand(B, M, 0).cuda(0)  # First tensor with middle dimension 0
    t2 = torch.rand(B, 0, N).cuda(0)  # Second tensor with initial dimension 0
    return (t1, t2)  # Return tuple of input tensors

