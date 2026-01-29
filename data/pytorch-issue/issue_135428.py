# torch.rand(5, 2, dtype=torch.complex64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Normalize complex tensor and its real component separately
        norm_complex = F.normalize(x, p=196)
        norm_real = F.normalize(x.real, p=196)
        real_part_complex = norm_complex.real
        
        # Compare real components using torch.allclose (GPU/real behavior)
        # Returns True if consistent with GPU/real case (no NaNs)
        are_close = torch.allclose(real_part_complex, norm_real, atol=1e-8)
        return torch.tensor(are_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 2, dtype=torch.complex64)

