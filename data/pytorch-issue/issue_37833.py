# torch.rand(1, 128, 201, dtype=torch.float32)  # Inferred input shape and dtype from error indices [0,95,19]
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Core operation causing inconsistency across CPU architectures
        return torch.log10(x)

def my_model_function():
    # Returns the model instance that demonstrates log10 inconsistency
    return MyModel()

def GetInput():
    # Generate random input matching expected 3D audio spectrogram shape (B,C,F)
    return torch.rand(1, 128, 201, dtype=torch.float32)

