# torch.rand(B, H, W, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Permute to create non-contiguous tensor (triggering CPU parallel copy on CUDA transfer)
        permuted = x.permute(0, 3, 1, 2)
        # Move to CUDA which forces contiguous copy if tensor is non-contiguous
        return permuted.cuda()

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape (B, H, W, C) from the issue's numpy array
    return torch.rand(8, 224, 224, 3, dtype=torch.float32)

