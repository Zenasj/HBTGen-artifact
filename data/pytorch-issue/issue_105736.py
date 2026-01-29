# (torch.rand(B, C, dtype=torch.float32), torch.tensor([-1, 2], dtype=torch.long))
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        y, t = inputs
        # Compute eager mode loss
        loss_eager = F.cross_entropy(y, t, ignore_index=-1)
        # Compute vmap version (which may fail due to the bug)
        try:
            loss_vmap = torch.vmap(F.cross_entropy)(y, t, ignore_index=-1)
        except:
            loss_vmap = None  # Capture errors as None
        # Return comparison result as a tensor
        if loss_vmap is None:
            return torch.tensor(-1.0)  # Indicates vmap failed
        else:
            return torch.tensor(1.0 if torch.allclose(loss_eager, loss_vmap) else 0.0)

def my_model_function():
    return MyModel()

def GetInput():
    B, C = 2, 3  # Batch size and class count from the example
    y = torch.rand(B, C, dtype=torch.float32)
    t = torch.tensor([-1, 2], dtype=torch.long)  # Contains ignore_index=-1 and valid target
    return (y, t)

