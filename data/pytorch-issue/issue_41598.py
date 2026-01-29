# torch.rand(B, 3, 800, 1200, dtype=torch.float32)  # Input shape inferred from object detection task (batch, channels, H, W)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified backbone mimicking Faster R-CNN structure (ResNet-50-like layers)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Dummy head layer to demonstrate forward pass
        self.fc = nn.Linear(64 * 200 * 300, 1)  # Assuming downsampled dimensions after backbone

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Flatten for dummy FC layer (for demonstration purposes)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x  # Dummy output to match forward requirements

def my_model_function():
    # Initialize model with CPU weights (no CUDA)
    model = MyModel()
    model.to(torch.device("cpu"))
    return model

def GetInput():
    # Generate random input tensor matching expected shape
    return torch.rand(2, 3, 800, 1200, dtype=torch.float32)  # Batch size 2, RGB images (800x1200 common for detection)

