# torch.rand(B, L, 100, dtype=torch.float32)  # B: batch, L: variable length dimension (columns)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_channels, 17)  # Output classes

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert to (batch, channels, length)
        x = self.conv1(x)
        x = x.transpose(1, 2)  # Revert to (batch, length, channels)
        x = self.relu(x)
        x = self.fc(x)
        return x

def my_model_function():
    # Initialize the model with parameters from the original issue's Conv1DModel
    return MyModel(in_channels=100, out_channels=100, kernel_size=1, padding=0)

def GetInput():
    # Generate random input with dynamic batch and length dimensions
    batch_size = torch.randint(100, 300, (1,)).item()
    length = torch.randint(200, 400, (1,)).item()
    return torch.rand(batch_size, length, 100, dtype=torch.float32)

