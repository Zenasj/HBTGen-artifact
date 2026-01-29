# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape is 2x3x8x8 as per the original code
import torch
import torch.nn as nn

class MyModel(nn.Module):  # Renamed from PreprocessAndCalculateModel
    def __init__(self):
        super().__init__()
        self.channel_shuffle = nn.ChannelShuffle(groups=3)
    
    def forward(self, x):
        output = self.channel_shuffle(x)
        return output

def my_model_function():
    return MyModel()  # Returns the model instance with ChannelShuffle layer

def GetInput():
    return torch.rand(2, 3, 8, 8)  # Matches the original input dimensions (B=2, C=3, H=8, W=8)

