# torch.rand(5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Method 1: torch.topk (could return any index in case of ties)
        topk_indices = torch.topk(x, k=1, largest=True).indices
        
        # Method 2: First occurrence of maximum value (deterministic)
        max_val = x.max()
        first_occurrence = (x == max_val).nonzero(as_tuple=True)[0][0]
        
        # Return True if both indices match, else False
        return torch.eq(topk_indices[0], first_occurrence).unsqueeze(0).float()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, dtype=torch.float32)

