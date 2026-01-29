# torch.rand(28, 28, dtype=torch.float32, device="cuda")  # inferred input shape and device
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 1)  # Matches the original Net structure

    def forward(self, x):
        return self.fc1(x)

def my_model_function():
    # Returns initialized model on CUDA device
    model = MyModel()
    model.to("cuda")  # Explicitly move to CUDA as in the repro script
    return model

def GetInput():
    # Generates input tensor matching CUDA device and shape
    return torch.rand(28, 28, device="cuda", dtype=torch.float32)

