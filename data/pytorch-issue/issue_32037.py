# torch.rand(B, 3, H, W, dtype=torch.float32)  # B=2, H/W range 200-400 (e.g., 300x300)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    import random
    height = random.randint(200, 400)
    width = random.randint(200, 400)
    return torch.rand(2, 3, height, width, dtype=torch.float32)

