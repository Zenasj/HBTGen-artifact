import torch
import math

# torch.rand((), dtype=torch.float32)
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # PyTorch's pow retains tensor structure and gradient
        torch_out = torch.pow(x, 3)
        # math.pow converts tensor to scalar (float), losing gradient
        math_out = math.pow(x, 3)
        # Return True if math_out is a scalar (float), indicating the footgun scenario
        is_scalar = isinstance(math_out, float)
        return torch.tensor(is_scalar, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), requires_grad=True)

