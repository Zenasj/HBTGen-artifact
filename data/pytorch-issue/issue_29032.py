# torch.rand(B, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.observer = torch.quantization.MinMaxObserver()
        self.observer_ = torch.quantization.MinMaxObserver()
    
    def forward(self, x):
        self.observer(x)
        self.observer_(x)
        return x  # Return input to satisfy forward contract

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 32)  # Matches input shape from the original example (B=1, features=32)

