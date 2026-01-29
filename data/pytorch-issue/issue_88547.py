# torch.randint(0, 2, (), dtype=torch.int64)  # shape: (), dtype: torch.int64
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute on CPU
        cpu_x = x.to('cpu')
        cpu_result = torch.bitwise_right_shift(cpu_x, 64)
        # Compute on CUDA if available
        if torch.cuda.is_available():
            cuda_x = x.to('cuda')
            cuda_result = torch.bitwise_right_shift(cuda_x, 64).to('cpu')
        else:
            return torch.tensor(False, dtype=torch.bool)  # No CUDA, no discrepancy possible
        # Compare results and return as tensor
        return torch.tensor(not torch.allclose(cpu_result, cuda_result), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(1, dtype=torch.int64)

