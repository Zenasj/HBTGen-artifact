# torch.rand(B, 3*2*2, 12, dtype=torch.float32)  # Input shape for MyModel

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fold = nn.Fold(output_size=(4,5), kernel_size=(2,2))
        
    def forward(self, x):
        return self.fold(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((1, 3*2*2, 12), dtype=torch.float32)

