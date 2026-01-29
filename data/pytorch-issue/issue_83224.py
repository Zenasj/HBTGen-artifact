# torch.rand(B, C, L, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, padding=-1, stride=50, return_indices=True)
    
    def forward(self, x):
        try:
            mod_out, mod_indices = self.pool(x)
            func_out, func_indices = F.max_pool1d(
                x, kernel_size=2, padding=-1, stride=50, return_indices=True
            )
            output_match = torch.allclose(mod_out, func_out, atol=1e-5)
            indices_match = (mod_indices == func_indices).all()
            return torch.tensor([output_match and indices_match], dtype=torch.bool)
        except RuntimeError:
            return torch.tensor([False], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 49, dtype=torch.float32)

