# torch.rand(1, 1, 1, 1, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 524288.0  # Scalar value from the example

    def forward(self, x):
        # Division using scalar (original problematic approach)
        out_scalar = x / self.scale
        # Division using tensor (workaround)
        scale_tensor = torch.tensor([self.scale], device=x.device, dtype=torch.float32)
        out_tensor = x / scale_tensor
        # Compare the outputs (cast to float32 for accurate comparison)
        diff = torch.abs(out_scalar.float() - out_tensor)
        # Return True if they are within a small threshold (1e-5)
        return torch.all(diff < 1e-5)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float16) * 3388.0  # Matches example's magnitude

