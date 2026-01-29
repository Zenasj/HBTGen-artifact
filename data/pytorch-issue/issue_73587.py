# torch.rand(3, 4, 4, dtype=torch.float32).cuda()  # Inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduce the indexing operation that triggers the bug in deterministic mode
        indices = torch.arange(2, device=x.device)  # Ensure indices match input device
        x[indices] = 1.0  # Fails in deterministic mode with specific shape/stride combinations
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Create input matching the failing scenario (3x4x4 float32 CUDA tensor)
    return torch.rand(3, 4, 4, dtype=torch.float32).cuda()

