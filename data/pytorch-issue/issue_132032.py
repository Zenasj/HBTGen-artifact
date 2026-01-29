# torch.rand(B, C, H, W, dtype=torch.float16)  # Input shape: (1, 3, 64, 64) in FP16 to trigger Winograd FP16 path
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d with kernel_size=3 and FP16 to trigger problematic Winograd codepath
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Returns a model instance using FP16 convolution (triggers ARM Compute Library code)
    model = MyModel()
    model.half()  # Ensure model uses float16 to hit the faulty FP16 codepath
    return model

def GetInput():
    # Generates 1x3x64x64 input tensor in FP16 matching model's requirements
    return torch.rand(1, 3, 64, 64, dtype=torch.float16)

