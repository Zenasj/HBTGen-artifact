# torch.rand(2, 3, 14, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Problematic MPS path: permute + slice + sqrt (may produce NaNs on MPS)
        mps_path = x.permute(1, 0, 2)[..., 3].sqrt()
        # Correct alternative: transpose + sqrt (works correctly on all backends)
        correct_path = x[..., 3].sqrt().transpose(0, 1)
        # Compare outputs using numerical tolerance
        return torch.tensor([torch.allclose(mps_path, correct_path, atol=1e-4, rtol=1e-5)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 14, dtype=torch.float32)

