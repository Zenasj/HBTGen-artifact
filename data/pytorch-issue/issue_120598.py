# torch.rand(3, 3, dtype=torch.float32, device='cuda')  # Input shape: (3,3) on CUDA
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
    
    def forward(self, x):
        # Replicate the original issue's stream operations
        with torch.cuda.stream(self.stream1):
            x = x + 2.0
        self.stream2.wait_stream(self.stream1)
        with torch.cuda.stream(self.stream2):
            x = x + 3.0
        return x

def my_model_function():
    # Returns model instance with internal streams
    return MyModel()

def GetInput():
    # Returns valid input tensor matching model's requirements
    return torch.rand(3, 3, dtype=torch.float32, device='cuda')

