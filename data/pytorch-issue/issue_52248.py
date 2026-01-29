# torch.rand(10, dtype=torch.float32, requires_grad=True)

import torch
from torch import nn

class NaNWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f1, f2, mask_fn=torch.isfinite):
        x_1 = x.detach().clone().requires_grad_(True)
        x_2 = x.detach().clone().requires_grad_(True)
        
        with torch.enable_grad():
            y_1 = f1(x_1)
            y_2 = f2(x_2)
        
        mask = mask_fn(y_1)
        
        ctx.save_for_backward(mask)
        ctx.x_1 = x_1
        ctx.x_2 = x_2
        ctx.y_1 = y_1
        ctx.y_2 = y_2
        
        return torch.where(mask, y_1, y_2)

    @staticmethod
    def backward(ctx, gout):
        mask, = ctx.saved_tensors
        torch.autograd.backward([ctx.y_1, ctx.y_2], [gout, gout])
        gin = torch.where(mask, ctx.x_1.grad, ctx.x_2.grad)
        return gin, None, None, None

nan_where = NaNWhere.apply

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def f1(self, x):
        return x.sqrt()
    
    def f2(self, x):
        return (-x).sqrt()
    
    def forward(self, x):
        return nan_where(x, self.f1, self.f2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.linspace(-5, 5, 10, requires_grad=True)

