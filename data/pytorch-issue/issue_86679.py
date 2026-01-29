# torch.rand(1, 4, 32, 32, dtype=torch.bfloat16) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Cast to float32 for upsample, then cast back to bfloat16
        x = x.to(dtype=torch.float32)
        x = self.upsample(x)
        x = x.to(dtype=torch.bfloat16)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 4, 32, 32, dtype=torch.bfloat16, device="cuda")

