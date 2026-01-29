# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Scalar parameter to trigger 0D tensor gradients
        self.param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        # Forward pass that uses the scalar parameter to create 0D gradients
        return self.param * x  # Element-wise multiplication with scalar

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape that matches the model's requirements (1x1x1x1 tensor)
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

