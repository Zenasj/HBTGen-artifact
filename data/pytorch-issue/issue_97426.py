import torch
import torch.nn as nn

# torch.rand(B, C, L, dtype=torch.float)  # Input shape inferred as (7, 3, 20)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Residual blocks with group norm (as per error context)
        self.groups = nn.Sequential(
            # First group of Res1dDownSample modules
            nn.Sequential(
                nn.Conv1d(3, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),  # 8 groups (example based on logs)
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
            ),
            # Second group with downsampling
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(16, 128),
            ),
        )
        # Lateral layers (from 'lateral' modules in logs)
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 64, kernel_size=1),
                nn.GroupNorm(32, 64),
            ),
            nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=1),
                nn.GroupNorm(16, 32),
            ),
        ])
        # Output layers
        self.output = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
        )

    def forward(self, x):
        x = self.groups(x)
        for layer in self.lateral:
            x = layer(x)
        return self.output(x)

def my_model_function():
    # Initialize model with example parameters (weights inferred from logs)
    model = MyModel()
    return model

def GetInput():
    # Generate input matching (B=7, C=3, L=20) from error logs
    return torch.rand(7, 3, 20, dtype=torch.float)

