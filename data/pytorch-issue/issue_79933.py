# torch.rand(1000000, 100, dtype=torch.float32, device='cuda:0')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare direct dtype conversion during transfer (operation2) vs two-step conversion (operation3)
        # Returns both outputs for external memory property inspection
        out2 = x.to('cpu', dtype=torch.float16, non_blocking=True)  # Direct conversion path (problematic)
        out3 = x.to(dtype=torch.float16).to('cpu', non_blocking=True)  # Two-step conversion (correct pinned memory)
        return out2, out3  # Return both for comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a CUDA tensor matching the input expected by MyModel
    return torch.rand(1000000, 100, dtype=torch.float32, device='cuda:0')

