# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.channel_shuffle = nn.ChannelShuffle(groups=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)  # Matches YOLOv5 scenario
        
    def forward(self, x):
        # Apply ChannelShuffle and MaxPool2d sequentially
        x = self.channel_shuffle(x)
        x = self.max_pool(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape matching original ChannelShuffle example
    return torch.rand(1, 32, 128, 128, dtype=torch.float32)

