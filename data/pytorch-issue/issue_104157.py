# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn
import warnings

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)

    def forward(self, x):
        return self.upsample(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((2, 2, 1, 1), dtype=torch.float32, requires_grad=True).cuda()

def get_grads(dtype, size):
    input = torch.randn((2, 2, 1, 1), dtype=dtype, requires_grad=True).cuda()
    output = interpolate(input, size=size, mode='bicubic', align_corners=True)
    grad_outputs = [torch.ones_like(output)]
    grads = torch.autograd.grad([output], [input], grad_outputs)
    return grads[0]

def interpolate(input: torch.Tensor, size: int, mode: str, align_corners: bool):
    if input.dtype == torch.float16 and mode in ('linear', 'bilinear', 'bicubic', 'trilinear', 'area'):
        warnings.warn(f"Using float16 for the backward pass of upsample mode `{mode}` can cause precision issues. If the result exceeds 2048, it will be truncated. For more details, refer to the discussion at https://github.com/pytorch/pytorch/issues/104157.", UserWarning, stacklevel=2)
    return torch.nn.functional.interpolate(input, size=size, mode=mode, align_corners=align_corners)

