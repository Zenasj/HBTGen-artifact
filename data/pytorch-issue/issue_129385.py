import torch
import torch.nn as nn

# torch.rand(1, 1, 1024, 1024, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dummy convolution layer to satisfy the model structure
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Initialize the model with dummy weights
    model = MyModel()
    return model

def GetInput():
    # Generate input matching the expected 4D tensor shape
    return torch.randn(1, 1, 1024, 1024, device='cuda')

