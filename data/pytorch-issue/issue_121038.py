# torch.rand(B, 36, 1024, 1024, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder architecture based on input shape inference
        self.conv1 = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 1024 * 1024, 10)  # Example output dimension

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Initialize model with required configuration
    model = MyModel()
    return model

def GetInput():
    # Generate input with dynamic batch dimension (batch size 2 as per user's fix suggestion)
    return torch.rand(2, 36, 1024, 1024, dtype=torch.float32)

