# torch.rand(1, 3, 5, 5, dtype=torch.float64)
import numpy as np
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=(2, 2),
            ceil_mode=False
        )
    
    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Create a transposed numpy array to produce non-contiguous input tensor
    shape = (1, 5, 5, 3)
    arr = np.random.randn(*shape).transpose(0, 3, 1, 2) * 10
    return torch.from_numpy(arr).to(dtype=torch.float64)

