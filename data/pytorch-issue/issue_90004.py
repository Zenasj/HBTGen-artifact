# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

def not_close_error_metas(t1, t2, rtol=1e-5, atol=1e-8):
    """Mock implementation of the comparison function returning error metadata."""
    errors = []
    if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
        errors.append("Outputs differ beyond tolerance")
    return errors

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modelA = nn.Sequential(
            nn.Conv2d(3, 6, 3),          # Model A: no padding
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 30 * 30, 10)   # 32-2=30 after kernel 3
        )
        self.modelB = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1),  # Model B: same padding
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 32 * 32, 10)    # 32x32 preserved with padding
        )

    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        errors = not_close_error_metas(outA, outB)
        # Return boolean tensor indicating equality
        return torch.tensor(not bool(errors), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 32, 32, dtype=torch.float32)

