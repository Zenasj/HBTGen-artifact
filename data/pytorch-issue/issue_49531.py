# torch.rand(1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.randn(2, 2, device='cuda'))
        self.register_buffer('b', torch.randn(2, 3, device='cpu'))

    def forward(self, x):
        c = self.a @ self.b  # Perform matmul between different devices
        try:
            # Attempt an operation that should fail if CUDA context is corrupted
            test = torch.tensor([1], device='cuda')
            return torch.tensor([1], dtype=torch.float32)  # Success
        except RuntimeError:
            return torch.tensor([0], dtype=torch.float32)  # Failure

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1)  # Dummy input, not used in forward

