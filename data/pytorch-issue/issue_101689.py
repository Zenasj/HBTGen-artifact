import torch
import torch.nn as nn

# torch.rand(B, 4, 64, 64, dtype=torch.float32)  # Inferred input shape based on typical UNet latent diffusion models
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 64, 3, padding=1)
        self.checkpoint_used = True  # Flag to toggle checkpoint usage
    
    def forward(self, x):
        def func(x):
            return self.conv(x)
        
        params = list(self.conv.parameters())
        return checkpoint(func, (x,), params, flag=self.checkpoint_used)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input tensor matching UNet's expected shape
    return torch.rand(2, 4, 64, 64, dtype=torch.float32)

# Checkpoint function causing ONNX export incompatibility (from user's provided code)
def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

