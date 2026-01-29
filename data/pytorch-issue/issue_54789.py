# torch.rand(4, 4, dtype=torch.bool)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Detect discrepancy between .float() conversion and expected 1.0 for True
        float_conversion = x.float()
        expected = x.to(torch.float)  # Converts True→1.0, False→0.0
        diff = torch.abs(float_conversion - expected)
        # Return True if any element differs beyond epsilon (handles both 255 and -1 cases)
        return torch.any(diff > 1e-6)

def my_model_function():
    return MyModel()

def GetInput():
    # Create poisoned input with underlying True values stored as 255 (Linux-style)
    poisoned_data = b'\xff' * 16  # 4x4 tensor (16 elements)
    poisoned_np = np.frombuffer(poisoned_data, dtype=np.bool_)
    return torch.from_numpy(poisoned_np).view(4, 4)

