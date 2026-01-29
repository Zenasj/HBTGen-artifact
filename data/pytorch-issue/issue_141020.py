# torch.rand(B, C, H, W, dtype=torch.float32)  # Example shape: (2, 3, 224, 224)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN structure for compatibility with distributed training scenarios
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 10)  # Adjusted for input size (224x224)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape for MyModel
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

