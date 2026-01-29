# torch.rand(2, 16, 768, 1152, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a simple convolution layer with channels_last memory format
        self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Ensure weights are initialized in channels_last format
        self.conv.weight.data = self.conv.weight.data.contiguous(memory_format=torch.channels_last)
        
    def forward(self, x):
        # Apply convolution to input tensor (channels_last format expected)
        return self.conv(x)

def my_model_function():
    # Return model instance with channels_last weights
    return MyModel()

def GetInput():
    # Generate input tensor matching the reported issue's shape and memory format
    input_tensor = torch.rand(2, 16, 768, 1152, dtype=torch.float32).to(memory_format=torch.channels_last)
    return input_tensor

