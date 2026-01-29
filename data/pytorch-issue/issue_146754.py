# torch.rand(1, 1024, 1024, dtype=torch.float32)  # Test case for N=1024
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.linalg.inv(x)

def my_model_function():
    return MyModel()

def GetInput():
    n = 1024  # Test case for N>1024
    torch.manual_seed(42)
    A = torch.randn(1, n, n, dtype=torch.float32)
    identity = torch.eye(n, dtype=torch.float32).unsqueeze(0)  # (1, N, N)
    A += n * identity  # Ensure invertibility via scaled identity addition
    return A

