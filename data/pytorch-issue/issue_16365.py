# torch.rand(65536, 2, 3, dtype=torch.float32, device='cuda')  # Inferred input shape for batch_size=256*256
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, theta):
        # Reproduces the bug scenario using affine_grid with batch_size=65536
        output_size = (theta.size(0), 1, 2, 2)  # Matches the original issue's tensor size
        return F.affine_grid(theta, output_size)

def my_model_function():
    # Returns the model instance causing the reported error
    return MyModel()

def GetInput():
    # Generates a random input tensor that triggers the CUDA grid size bug
    batch_size = 256 * 256  # 65536 - exceeds CUDA grid's y dimension limit
    return torch.rand(batch_size, 2, 3, dtype=torch.float32, device='cuda')

