# torch.rand(B, 4, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.channel_shuffle = nn.ChannelShuffle(2)  # groups=2 as per minified example
        
    def forward(self, x):
        return self.channel_shuffle(x)
    
def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces input shape from error logs (1,4,2,2) and minified example
    return torch.randn(1, 4, 2, 2, dtype=torch.float32)

