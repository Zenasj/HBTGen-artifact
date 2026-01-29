# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MinstSteerableCNN_simple(nn.Module):
    def __init__(self, num_classes, tranNum):
        super(MinstSteerableCNN_simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.tranNum = tranNum
        self.filter = None
        self.bias = None

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.filter = torch.randn(64, 32, 3, 3).to(self.conv2.weight.device)
            self.bias = torch.randn(64).to(self.conv2.bias.device)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MinstSteerableCNN_simple(num_classes=10, tranNum=4)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 64, 1, 28, 28  # Example batch size, channels, height, width
    return torch.rand(B, C, H, W, dtype=torch.float32)

