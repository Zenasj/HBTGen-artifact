# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Example input shape for image-like tensors
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        x_out = x + 1
        contig = x_out.contiguous()  # Creates an alias tensor
        return x_out, contig  # Both outputs point to the same underlying tensor

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Standard image input shape

