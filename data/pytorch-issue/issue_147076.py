# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm2d(16)  # Example: BatchNorm2d with 16 channels
        self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    model.eval()  # Ensure the model is in evaluation mode
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean = module.running_mean.detach()
            module.running_var = module.running_var.detach()
    model = model.to(torch.float32)  # Convert the entire model to float32
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 16, 32, 32  # Example: batch size 1, 16 channels, 32x32 image
    return torch.rand(B, C, H, W, dtype=torch.float32).cuda()

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

