# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Segfault occurs here when device 'cuda' is used on non-CUDA machine
        self.generator = torch.Generator(device='cuda')  

    def forward(self, x):
        return x  # Dummy forward pass to satisfy module requirements

def my_model_function():
    return MyModel()  # Returns model that triggers the described bug

def GetInput():
    # Returns a standard input tensor matching the expected shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

