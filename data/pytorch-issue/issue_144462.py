# torch.rand(1, 1, 1, 1, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.const_pad = nn.ConstantPad2d(padding=(10, 10, 0, 0), value=0.0)  # Left/right pad 10, top/bottom 0

    def forward(self, x):
        # Apply both padding methods
        func_pad = F.pad(x, pad=(10, 10), mode='constant', value=0.0)
        const_pad = self.const_pad(x)
        
        # Check if padded regions (except the central input area) are zero (correct padding)
        def is_correct(padded):
            # Check first 10 and last 10 elements in the last dimension
            return torch.all(padded[..., :10] == 0) and torch.all(padded[..., -10:] == 0)
        
        func_correct = is_correct(func_pad)
        const_correct = is_correct(const_pad)
        
        # Return True only if both methods are correct
        return func_correct and const_correct

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 1, 1, dtype=torch.float32, device='cuda')

