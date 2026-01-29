# torch.rand(1, 4, 4, 4, dtype=torch.float)
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMaxPool2dWithIndices(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output, indices = F.adaptive_max_pool2d(input, (3, 1), return_indices=True)
        ctx.save_for_backward(input, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, indices = ctx.saved_tensors
        # Force indices to 5 (as in the issue's example) to trigger overflow
        indices.fill_(5)
        return torch.ops.aten.adaptive_max_pool2d_backward(grad_output, input, indices)

class MyModel(nn.Module):
    def forward(self, x):
        return AdaptiveMaxPool2dWithIndices.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Replicate the input tensor from the issue (shape and value)
    return torch.full((1, 4, 4, 4), 7.29222, dtype=torch.float)

