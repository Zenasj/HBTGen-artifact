# torch.rand(B, 1, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.seq(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Required for quantization preparation
    return model

def GetInput():
    # Returns a random input tensor with shape (batch_size, 1)
    return torch.rand(4, 1)  # Batch size 4 is arbitrary but consistent

