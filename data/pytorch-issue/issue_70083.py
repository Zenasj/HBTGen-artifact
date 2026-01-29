# torch.rand(B, 1, dtype=torch.float32)
import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1))
        self.b = torch.nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return F.linear(x, self.w, self.b)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor with shape (batch_size, 1)
    return torch.rand(2, 1, dtype=torch.float32)

