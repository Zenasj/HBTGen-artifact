import torch
import torch.nn as nn

# torch.rand(B, 517, 5, dtype=torch.float64)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1, dtype=torch.float64)  # Explicit dtype for reproducibility

    def forward(self, x):
        return self.fc(x).squeeze(dim=-1)

def my_model_function():
    # Initialize model with float64 and move to CUDA (bug scenario)
    model = MyModel()
    return model.to(device='cuda', dtype=torch.float64)

def GetInput():
    # Reproduces the input scaling from the original issue
    return (torch.randn(5, 517, 5, dtype=torch.float64, device='cuda') - 0.5) * 2

