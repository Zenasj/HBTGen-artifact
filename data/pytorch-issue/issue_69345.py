# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 224 * 224, 10)  # Example output layer for classification

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected shape (B, 3, 224, 224)
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

