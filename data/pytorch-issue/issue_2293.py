# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Model with default BatchNorm (momentum=0.1) and fixed BatchNorm (momentum=0.0)
        # to compare the effect on running averages during training with LR=0
        self.model_default = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, momentum=0.1),  # Default momentum
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
        self.model_fixed = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, momentum=0.0),  # Fixed momentum to prevent running stat updates
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
    
    def forward(self, x):
        # Compare outputs between default and fixed models
        out_default = self.model_default(x)
        out_fixed = self.model_fixed(x)
        # Return difference tensor to indicate discrepancy (non-zero values mean differences)
        return out_default - out_fixed

def my_model_function():
    # Returns a model instance with both submodules for comparison
    return MyModel()

def GetInput():
    # Generate random input matching expected shape (3 channels, 32x32 image)
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

