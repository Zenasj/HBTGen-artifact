# torch.rand(1, 1, 5, 5, dtype=torch.float)  # Input shape: (B, C, H, W)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize the grid as a parameter with identity affine transformation
        theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # Identity matrix
        grid_size = (1, 1, 5, 5)  # Matches input dimensions from example
        self.grid = nn.Parameter(F.affine_grid(theta, grid_size, align_corners=True))
    
    def forward(self, image):
        return F.grid_sample(
            image,
            self.grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 5, 5, dtype=torch.float)

