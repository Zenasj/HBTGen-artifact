# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(64 * 56 * 56, 10)
        # Additional layer to simulate "extra loss" scenario mentioned in the issue
        self.fc_extra = nn.Linear(64 * 56 * 56, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        main_output = self.fc1(x)
        # Simulate additional loss path (common in multi-task setups)
        extra_output = self.fc_extra(x)
        return main_output, extra_output

def my_model_function():
    # Initialize model with DDP compatibility (no actual DDP here, just base model)
    model = MyModel()
    # Dummy initialization to ensure weights are present
    model(torch.randn(2, 3, 224, 224))
    return model

def GetInput():
    # Random input matching the model's expected input shape (B, C, H, W)
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

