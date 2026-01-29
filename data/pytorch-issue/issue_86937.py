# torch.rand(B, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute MPS result (may be incorrect due to bug)
        mps_result = torch.matmul(x, x)
        # Compute expected CPU result
        cpu_x = x.cpu()
        cpu_result = torch.matmul(cpu_x, cpu_x)
        # Return the element-wise difference between MPS and CPU results
        return mps_result.cpu() - cpu_result

def my_model_function():
    # Returns the model instance
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape
    return torch.rand(1, 2, 2, dtype=torch.float32)

