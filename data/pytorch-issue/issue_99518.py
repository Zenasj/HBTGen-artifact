# torch.rand(B, 1, L, dtype=torch.complex64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, dtype=torch.complex64)
        # NCCL incompatibility will manifest during distributed training
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    L = 16  # Input length (arbitrary example)
    return torch.rand(B, 1, L, dtype=torch.complex64)

