# torch.rand(10000, 10000, dtype=torch.double, device='cuda')
import torch
from torch import nn

class Functional1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, y0, *params):
        ctx.fcn = fcn
        ctx.save_for_backward(y0, *params)
        return y0  # Returns y0 as per original implementation

    @staticmethod
    def backward(ctx, grad_yout):
        yout, *params = ctx.saved_tensors
        with torch.enable_grad():
            params_copy = [p.clone().requires_grad_() for p in params]
            yfcn = ctx.fcn(yout, *params_copy)
        # Preserve original create_graph logic from the issue's code
        grad_params = torch.autograd.grad(
            yfcn, params_copy, grad_outputs=grad_yout,
            create_graph=torch.is_grad_enabled()
        )
        return (None, None, *grad_params)  # Gradients for fcn (None), y0 (None), then params

def fcn2(x, x2):
    return x + x2

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize parameter equivalent to y2 in the test case (zeros with grad)
        self.param = nn.Parameter(torch.zeros(10000, 10000, dtype=torch.double, device='cuda').requires_grad_())

    def forward(self, x):
        return Functional1.apply(fcn2, x, self.param)  # Apply Functional1 with fcn2 and parameters

def my_model_function():
    return MyModel()  # Return initialized model instance

def GetInput():
    return torch.rand(10000, 10000, dtype=torch.double, device='cuda')  # Match input shape/dtype from test

