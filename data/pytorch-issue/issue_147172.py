# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = nn.Linear(d_hid, d_hid)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x

def my_model_function():
    # Initialize with hidden size 1024 as per original script
    return MyModel(d_hid=1024)

def GetInput():
    # Batch size 2, feature dim 1024 (matches d_hid=1024)
    return torch.rand(2, 1024, dtype=torch.float32)

