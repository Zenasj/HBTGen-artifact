# torch.rand(1, 1, 3, 1, dtype=torch.int64)  # Inferred input shape after workaround
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Minimal identity model to process the tensor
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Create problematic numpy array (int32 with dtype.num=5)
    b = np.array([1, 0, 1], dtype=np.int32)
    # Apply workaround: cast to int (becomes int64 on 64-bit systems)
    fixed_b = b.astype(int)
    # Reshape to 4D tensor (B, C, H, W)
    return torch.from_numpy(fixed_b.reshape(1, 1, 3, 1)).to(torch.int64)

