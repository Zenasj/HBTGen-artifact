# torch.rand(1, 32, 192, 192, 192, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.SyncBatchNorm(num_features=32)  # Matches the model in the issue's test code

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Returns the SyncBatchNorm model instance as in the issue's test setup
    return MyModel()

def GetInput():
    # Creates 5D tensor with channels_last_3d format to trigger the reported slowdown
    data = torch.rand(1, 32, 192, 192, 192, dtype=torch.float32)
    data = data.contiguous(memory_format=torch.channels_last_3d)
    return data

