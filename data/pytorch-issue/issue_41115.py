# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Create CUDA quantized tensor to trigger pinned memory issue
        q_int = torch.randint(0, 100, (1, 2, 3), device="cuda", dtype=torch.int)
        q = torch._make_per_tensor_quantized_tensor(q_int, scale=0., zero_point=0)
        return x  # Return input as output (unused, but required for model interface)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)  # Matches the input expectation of the model's forward()

