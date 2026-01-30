import torch

class MyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.sin(x), torch.sin(y)
    
    @staticmethod
    def backward(ctx, gO_x, gO_y):
        (x, y) = ctx.saved_tensors
        return gO_x * torch.cos(x), gO_y * torch.cos(y)