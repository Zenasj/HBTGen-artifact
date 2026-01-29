# torch.rand(B, C, L, dtype=torch.bfloat16)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pad = nn.ReplicationPad1d(15)  # Assuming kernel_size=32 for PQMF filters
        # Example parameters: 4 subbands (C=4), filter length=32
        self.filters = nn.Parameter(torch.randn(1, 4, 32, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.pad(x)
        return F.conv1d(x, self.filters)

def my_model_function():
    model = MyModel()
    return model  # Already in BFloat16 via filters' initialization

def GetInput():
    # Input shape after interpolation (B, n_subbands, length * n_subbands)
    return torch.rand(1, 4, 40, dtype=torch.bfloat16)  # Example length 40

