# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.rand(32, 3, 3, 3, dtype=torch.float32)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        return torch._convolution(
            x, self.weight, None,  # bias
            [2, 2], [0, 0], [1, 1],  # stride, padding, dilation
            False, [0, 0], 1,  # transposed, output_padding, groups
            False, False, True  # benchmark, deterministic, cudnn_enabled
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, requires_grad=True)

