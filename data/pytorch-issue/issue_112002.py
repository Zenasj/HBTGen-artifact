import torch
from torch import nn

# torch.rand(B, 16, 64, dtype=torch.float16) for each of the three inputs (dq, dk, dv)
class StackTest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dq, dk, dv):
        ctx.save_for_backward(dq, dk, dv)
        return dq

    @staticmethod
    def backward(ctx, dout):
        dq, dk, dv = ctx.saved_tensors
        dqkv = torch.stack([dq, dk, dv], dim=1)
        return dq, None, None  # Returns input tensor causing the error

class MyModel(nn.Module):
    def forward(self, inputs):
        dq, dk, dv = inputs
        return StackTest.apply(dq, dk, dv)

def my_model_function():
    return MyModel()

def GetInput():
    B = 10
    dq = torch.randn(B, 16, 64, dtype=torch.float16, requires_grad=True).cuda()
    dk = torch.randn(B, 16, 64, dtype=torch.float16, requires_grad=True).cuda()
    dv = torch.randn(B, 16, 64, dtype=torch.float16, requires_grad=True).cuda()
    return (dq, dk, dv)

