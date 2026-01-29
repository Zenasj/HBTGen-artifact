# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest', align_corners=False):
        super(MyModel, self).__init__()
        self.mode = mode
        self.align = align_corners
        self.scale = scale_factor

    def forward(self, x):
        _, _, h, w = x.shape
        return nn.Upsample(size=(h * self.scale, w * self.scale), 
                          mode=self.mode, 
                          align_corners=self.align)(x)

def my_model_function():
    # Return an instance of MyModel with parameters from the issue's example
    return MyModel(mode='bilinear')  # Matches the user's export example in the issue

def GetInput():
    # Return a random input tensor with shape (1, 3, 64, 64)
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

