# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, momentum=None):
        super(MyModel, self).__init__()
        self.momentum = momentum  # Replicates the BatchNorm2d attribute causing the issue

    def forward(self, x):
        # Compute using original if-else logic (correct in eager mode)
        if self.momentum is None:
            exp_if = 0.0
        else:
            exp_if = self.momentum

        # Compute using ternary operator (problematic in TorchScript)
        exp_ter = 0.0 if self.momentum is None else self.momentum

        # Return True if outputs differ (indicates TorchScript bug)
        return torch.tensor(exp_if != exp_ter, dtype=torch.bool)

def my_model_function():
    # Initialize with momentum=None to replicate the original test case
    return MyModel(momentum=None)

def GetInput():
    # BatchNorm2d expects 4D input (B, C, H, W)
    return torch.rand(2, 3, 4, 4, dtype=torch.float32)

