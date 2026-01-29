# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.register_buffer('p', torch.tensor(4.0))  # Ensure p is treated as part of the model's state

    def forward(self, x):
        # Compute Lp norm using adaptive average pooling to handle dynamic input shapes
        powered = x.pow(self.p)
        pooled = F.adaptive_avg_pool2d(powered, (1, 1))  # Dynamically adjusts kernel size based on input
        result = pooled.pow(1.0 / self.p)
        return result

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the original dummy input shape (1,3,800,900) but works with any dynamic H/W
    return torch.rand(1, 3, 800, 900, dtype=torch.float32)

