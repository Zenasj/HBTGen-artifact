# torch.rand(1, dtype=torch.float32)
import torch
import torch.distributed as dist
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Reproduces distributed communication logic from the issue's run() function
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                x += 100
                dist.send(tensor=x, dst=1)
            else:
                dist.recv(tensor=x, src=0)
        return x

def my_model_function():
    # Returns the model instance (no weights required for distributed comms)
    return MyModel()

def GetInput():
    # Matches the input shape used in the original issue's code
    return torch.zeros(1, dtype=torch.float32)

