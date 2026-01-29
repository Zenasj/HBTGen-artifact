import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float16)  # Input shape (16, 64, 128, 128) based on minifier args
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution parameters inferred from backward call: kernel_size=3, stride=1, padding=1
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float16)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        conv_out = self.conv(x)
        relu_out = self.relu(conv_out)
        # Additional operations to match the Repro's forward path
        # (simplified to core convolution and activation)
        return relu_out

def my_model_function():
    # Initialize with float16 weights as seen in the minifier's args
    model = MyModel()
    return model

def GetInput():
    # Input shape from minifier args: (16, 64, 128, 128) with float16
    return torch.rand(16, 64, 128, 128, dtype=torch.float16, device="cuda")

