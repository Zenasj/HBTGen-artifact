# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize theta as float32 to avoid integer grid computation issue
        theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        theta = theta.view(1, 2, 3)
        self.theta = nn.Parameter(theta, requires_grad=False)  # Fixed identity matrix

    def forward(self, x):
        grid = F.affine_grid(self.theta, x.size())
        return F.grid_sample(x, grid)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns random tensor matching (B, C, H, W) = (1, 3, 10, 10) with float32 dtype
    return torch.rand(1, 3, 10, 10, dtype=torch.float32)

