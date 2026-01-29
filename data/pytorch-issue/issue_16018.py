# torch.rand(B, 25, 15, dtype=torch.float32)
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(25, 64, 7, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    np.random.seed(1034)  # Replicate the seed from the original code
    d_in, d_out, k = 25, 64, 7
    # Generate weights and bias as per original code
    w_np = np.random.random((k, d_in, d_out)).astype('float32')
    b_np = np.random.random((d_out,)).astype('float32')
    # Permute weight dimensions to (out_channels, in_channels, kernel_size)
    w = torch.as_tensor(w_np.transpose(2, 1, 0))
    b = torch.as_tensor(b_np)
    model.conv.weight = nn.Parameter(w)
    model.conv.bias = nn.Parameter(b)
    return model

def GetInput():
    B, L, d_in = 2, 10, 25
    original_input = torch.rand(B, L, d_in)
    x = original_input.transpose(-1, -2)  # Shape becomes (B, 25, 10)
    padded_x = F.pad(x, (2, 3), mode='constant', value=0)  # Pad to (B, 25, 15)
    return padded_x

