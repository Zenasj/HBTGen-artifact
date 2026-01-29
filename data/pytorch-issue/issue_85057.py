# torch.rand(2, 2, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Register problematic out tensor with mismatched shape as a buffer
        self.register_buffer('out', torch.ones(2, 2, 2, 2))

    def forward(self, x):
        # Reproduces the segmentation fault by using incompatible out tensor
        return F.normalize(x, out=self.out)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape (2,2) that triggers the error
    return torch.rand(2, 2, dtype=torch.float32)

