# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape is (2048, 256, 8, 8)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Core problematic upsample layer that triggers CUDA config error in specific PyTorch versions
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces input shape from failing test case (batch=2048, channels=256, 8x8 spatial)
    return torch.rand(2048, 256, 8, 8, dtype=torch.float32).cuda()

