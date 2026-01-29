# torch.rand(B, C, H, W, dtype=...)  # Input shape: (batch_size, channels, height, width)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.SyncBatchNorm(2, momentum=0.99)

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming batch size of 2, 2 channels, and variable height and width
    batch_size = 2
    channels = 2
    height = 300
    width = 300
    input_tensor = torch.rand(batch_size, channels, height, width, dtype=torch.float32)
    return input_tensor

