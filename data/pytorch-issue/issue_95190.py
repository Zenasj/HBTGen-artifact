# torch.rand(16, 1, 128, 128, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # In-place sigmoid causing version mismatch in AOTAutograd
        x.sigmoid_()
        return torch.clamp(x, min=1e-4, max=1-1e-4)

def my_model_function():
    return MyModel()

def GetInput():
    # Match shape/dtype from minifier and original issue
    return torch.rand(16, 1, 128, 128, dtype=torch.float32, 
                     device='cuda', requires_grad=True)

