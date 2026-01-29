# torch.rand(1, 4, 2, 2, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.channel_shuffle = nn.ChannelShuffle(2)

    def forward(self, x):
        return self.channel_shuffle(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 4, 2, 2, dtype=torch.float32)

