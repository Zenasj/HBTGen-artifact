# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Submodule name matches the qconfig_dict key from the issue
        self.ModuleName = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Additional layers for a complete model structure
        self.fc = nn.Linear(64 * 112 * 112, 10)

    def forward(self, x):
        x = self.ModuleName(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape comment and model's expected input
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

