# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (1, 3, 3, 3)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.InstanceNorm2d(3),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 3, 3, 3, requires_grad=True)

