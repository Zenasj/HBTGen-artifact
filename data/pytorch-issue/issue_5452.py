# torch.rand(1, 3, 10, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)  # Matches input channel count and preserves spatial dims
        self.logsoftmax = nn.LogSoftmax(dim=1)  # Output log probabilities for NLLLoss2d

    def forward(self, x):
        x = self.conv(x)
        return self.logsoftmax(x)

def my_model_function():
    # Initialize model with default settings (no weights provided as issue doesn't specify)
    model = MyModel()
    return model

def GetInput():
    # Generate random input tensor matching the expected shape (B=1, C=3, H=10, W=10)
    return torch.rand(1, 3, 10, 10, dtype=torch.float32)

