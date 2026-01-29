# torch.rand(1, 8, 8, 8, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_unpool3d = nn.MaxUnpool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Use indices of ones (as per original code's behavior)
        indices = torch.ones_like(x).long()
        return self.max_unpool3d(x, indices=indices)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 8, 8, 8, 1, dtype=torch.float32)

