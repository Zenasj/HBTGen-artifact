# torch.rand(2, 3, 32, 32, dtype=torch.float32)  # Inferred input shape based on 2-GPU setup
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified model structure to align with distributed training context
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)  # Matches 32x32 input after pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)

def my_model_function():
    # Returns a simple model instance for distributed testing
    return MyModel()

def GetInput():
    # Generates input matching model's expected dimensions
    return torch.rand(2, 3, 32, 32, dtype=torch.float32).cuda()  # Matches 2-GPU setup

