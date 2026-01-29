import torch
import numpy as np
from torch import nn

# torch.rand(1, 1, 128, 128, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Dummy model to demonstrate input compatibility
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Create input with uint16 -> int16 conversion as workaround
    arr = np.ones((128, 128, 1), dtype=np.uint16)
    arr_int16 = arr.astype(np.int16)
    tensor = torch.tensor(arr_int16, dtype=torch.float32)
    # Reshape to (B, C, H, W) format
    return tensor.unsqueeze(0).permute(0, 3, 1, 2)

