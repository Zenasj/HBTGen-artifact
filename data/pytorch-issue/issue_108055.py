# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Assumed input shape based on common CNN use cases
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN architecture for image-like inputs (3 channels, 32x32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)  # After two pooling layers (32x8x8)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 → 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 → 8x8
        x = x.view(-1, 32 * 8 * 8)
        return self.fc(x)

def my_model_function():
    # Returns a default instance of MyModel
    return MyModel()

def GetInput():
    # Returns random input tensor matching the model's expected dimensions
    return torch.rand(2, 3, 32, 32, dtype=torch.float32)  # Batch=2, 3 channels, 32x32 resolution

