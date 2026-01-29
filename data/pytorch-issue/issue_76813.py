# torch.rand(B, C, H, W, dtype=torch.float32)  # Dummy input shape, not used by the model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Returns a tensor with incorrect dtype (int32 instead of expected int64)
        return torch.tensor([1], dtype=torch.int32)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a dummy input tensor (shape and dtype arbitrary since model ignores input)
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

