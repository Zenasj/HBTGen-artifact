# torch.rand(B, 1, 28, 28, dtype=torch.float)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Standard CNN architecture for MNIST (2 conv layers + FC)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 7x7 from 2x2 pooling steps

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 28→14 after first pool
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 14→7 after second pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten for FC layer
        return self.fc(x)

def my_model_function():
    # Returns initialized model instance
    return MyModel()

def GetInput():
    # Generate random MNIST-like input tensor
    batch_size = 4  # Arbitrary batch size
    return torch.rand(batch_size, 1, 28, 28, dtype=torch.float)

