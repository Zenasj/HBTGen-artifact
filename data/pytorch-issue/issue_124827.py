# torch.rand(1, 4, dtype=torch.float32)
import torch
from torch import nn

class CommOpGradientScaling(torch.autograd.Function):
    _compiled_autograd_should_lift = False  # Fix from issue comments
    
    @staticmethod
    def forward(ctx, input_tensor, scale_gradient_factor):
        ctx.scale_gradient_factor = scale_gradient_factor
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_output.mul_(ctx.scale_gradient_factor)
        return grad_output, None

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        
    def forward(self, x):
        x = CommOpGradientScaling.apply(x, 0.5)
        return self.fc1(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 4, requires_grad=True)

