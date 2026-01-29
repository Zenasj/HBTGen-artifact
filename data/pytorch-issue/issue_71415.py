# Input shape: a tuple of (input (N, C, D, H, W), grid (N, D_out, H_out, W_out, 3)), dtype=torch.double

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        input, grid = x
        return F.grid_sample(
            input,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

def my_model_function():
    return MyModel()

def GetInput():
    N = 1  # Simplified batch size for minimal example (original test uses N=100)
    C = 2
    D_in, H_in, W_in = 2, 30, 40  # Input spatial dimensions from first test case
    D_out, H_out, W_out = 4, 13, 13  # Grid output dimensions from test code
    input = torch.rand(
        N, C, D_in, H_in, W_in,
        dtype=torch.double,
        requires_grad=False  # Matches original test input requirements
    )
    grid = 2.0 * torch.rand(
        N, D_out, H_out, W_out, 3,
        dtype=torch.double
    ) - 1.0
    grid.requires_grad_(True)  # Matches grid's gradient requirements in test
    return (input, grid)

