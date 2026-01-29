# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (batch_size, channels, height, width)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.zero_pad = nn.ZeroPad2d(padding=(1, 2))
        self.constant_pad = nn.ConstantPad3d(padding=1, value=0)

    def forward(self, x):
        # Apply ZeroPad2d
        y_zero_pad = self.zero_pad(x)
        
        # Reshape the input to 5D for ConstantPad3d
        x_reshaped = x.unsqueeze(0).unsqueeze(0)  # Add two dimensions
        y_constant_pad = self.constant_pad(x_reshaped)
        y_constant_pad = y_constant_pad.squeeze(0).squeeze(0)  # Remove the added dimensions
        
        return y_zero_pad, y_constant_pad

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming input shape (B, C, H, W)
    B, C, H, W = 3, 3, 32, 32
    return torch.randn(B, C, H, W, dtype=torch.float32)

