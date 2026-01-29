# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        try:
            # Legacy constructor (problematic path)
            legacy = torch.Tensor(x, device='cpu')
        except RuntimeError:
            # Legacy path failed (expected error)
            return torch.tensor(0.0)
        # Correct construction using recommended method
        correct = x.to('cpu').clone().detach()
        # Compare outputs (0.0 if mismatch, 1.0 if match)
        return torch.tensor(1.0) if torch.allclose(legacy, correct, atol=1e-5) else torch.tensor(0.0)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor matching expected input shape
    return torch.rand(1, 1, 1, 1, device='cuda', dtype=torch.float32)

