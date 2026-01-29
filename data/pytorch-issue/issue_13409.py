# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        mask = torch.rand_like(x).to(torch.bool)  # Generate mask matching input shape (BoolTensor for modern PyTorch)
        return x.masked_fill(mask, 1.0)  # Correct usage of masked_fill (out-of-place)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor with arbitrary dimensions (B=2, C=3, H=4, W=5)
    return torch.rand(2, 3, 4, 5)

