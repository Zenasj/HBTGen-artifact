# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Indices must match the gather operation in the original example
        indices = torch.tensor([0, 1], device='cuda:0')
        self.register_buffer('indices', indices)  # Non-trainable buffer

    def forward(self, x):
        # Perform gather operation as in the issue's example
        return x.gather(0, self.indices)

def my_model_function():
    # Returns the model instance with fixed indices and CUDA device
    return MyModel()

def GetInput():
    # Returns input tensor matching the model's requirements
    return torch.rand(4, dtype=torch.float32, device='cuda:0', requires_grad=True)

