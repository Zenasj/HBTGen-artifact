# torch.rand(B, 32, dtype=torch.float16)  # Input shape: batch_size x 32
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Weight initialized as float8_e5m2 on CUDA (as per issue's example)
        self.register_buffer('weight', torch.randn(32, 32, device='cuda').to(torch.float8_e5m2))
    
    def forward(self, x):
        # Convert weight to input's dtype (float16 in example) before matmul
        weight = self.weight.to(x.dtype)
        return torch.mm(x, weight)

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces input from the original issue's example (1x32 float16 tensor on CUDA)
    return torch.randn(1, 32, dtype=torch.float16).cuda()

