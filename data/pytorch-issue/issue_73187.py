# torch.rand(B, C, D, H, W, dtype=torch.float64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Grid shape: (N, D, H, W, 3) for 3D coordinates
        self.grid = nn.Parameter(torch.full((1, 1, 1, 1, 3), 42.0, dtype=torch.float64))
        self.interpolation_mode = 0  # Nearest neighbor (matches issue's example)
        self.padding_mode = 0        # Zeros padding (matches issue's example)
        self.align_corners = True    # Matches issue's example configuration

    def forward(self, x):
        return torch.grid_sampler_3d(
            x,
            self.grid,
            self.interpolation_mode,
            self.padding_mode,
            self.align_corners
        )

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random 5D tensor matching the input shape (B,C,D,H,W)
    return torch.rand(1, 1, 1, 1, 1, dtype=torch.float64)

