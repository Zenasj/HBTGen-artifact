# torch.rand(B, C, L, dtype=torch.float32)
import torch
from torch import nn

class MaxPool1dOld(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # Compute output shape assuming stride=1, padding=0, dilation=1
        batch_size, channels, length = x.shape
        out_length = (length - self.kernel_size) + 1
        return torch.empty(
            batch_size,
            channels,
            out_length,
            dtype=x.dtype,
            device=x.device
        )

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.new_maxpool = nn.MaxPool1d(3)  # PR-affected version
        self.old_maxpool = MaxPool1dOld(3)  # Simulated pre-PR behavior

    def forward(self, x):
        old_out = self.old_maxpool(x)
        try:
            new_out = self.new_maxpool(x)
            return torch.allclose(old_out, new_out)  # Returns True if outputs match
        except RuntimeError as e:
            # New version errors (shape check failed), so return False
            return torch.tensor(False)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the problematic input shape from the PR description
    return torch.rand(17, 0, 50, dtype=torch.float32)

