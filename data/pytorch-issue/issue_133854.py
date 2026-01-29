# torch.rand(B, C, L, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

def torch_conv(x, weight):
    _, d_model, L = x.shape
    kernel_size = weight.shape[-1]
    y = F.conv1d(
        x,
        weight,
        bias=None,
        stride=1,
        padding=kernel_size - 1,
        groups=d_model,
    )
    y = y[..., :L]
    return y

class MyCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        with torch.enable_grad():  # Critical fix to enable grad tracking
            y = torch_conv(x, weight)
        ctx.save_for_backward(x, weight, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, y = ctx.saved_tensors
        # Use autograd to compute gradients automatically
        grad_x, grad_weight = torch.autograd.grad(
            y, (x, weight),
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
        )
        return grad_x, grad_weight

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Kernel size inferred as 3 (hl=3) based on padding logic
        self.weight = nn.Parameter(torch.randn(768, 1, 3, 
                                              dtype=torch.float32, 
                                              device='cuda'))

    def forward(self, x):
        return MyCustomFunction.apply(x, self.weight)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, L = 1, 768, 8192
    return torch.randn(B, C, L, 
                      dtype=torch.float32, 
                      device='cuda', 
                      requires_grad=True)

