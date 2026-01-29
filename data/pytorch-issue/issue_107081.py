# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Assumed input shape (B, C, H, W)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified model structure for reproducibility
        self.fc = nn.Linear(3 * 224 * 224, 10)  # Matches input shape's flattened dimensions

    def forward(self, x):
        # Flatten input tensor for linear layer
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    # Returns an instance of the model with standard initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the assumed input shape
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

