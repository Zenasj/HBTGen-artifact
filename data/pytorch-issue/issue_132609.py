# torch.rand(2, dtype=torch.float, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return CustomOp.apply(x)

def my_model_function():
    return MyModel()

class CustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i + i
        ctx.save_for_backward(i)  # Fixed: use method instead of assignment
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]  # Retrieve saved tensors correctly
        detached_inputs = input.detach()
        with torch.enable_grad():
            outputs = detached_inputs * detached_inputs
        output_tensors = []
        grad_tensors = []
        if outputs.requires_grad:  # Dynamo error trigger line
            output_tensors.append(outputs)
            grad_tensors.append(grad_output)
        torch.autograd.backward(output_tensors, grad_tensors)
        return detached_inputs.grad  # Correct attribute access

def GetInput():
    return torch.rand(2, dtype=torch.float, requires_grad=True)

