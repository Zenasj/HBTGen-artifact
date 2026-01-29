# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Conv layer matching smoke test patterns
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Linear layer to match output shape in smoke test (1000)
        self.fc = nn.Linear(16 * 224 * 224, 1000)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten for linear layer
        x = self.fc(x)
        return x

def my_model_function():
    # Initialize model with default parameters
    return MyModel()

def GetInput():
    # Generate input matching (B, C, H, W) = (1,3,224,224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

