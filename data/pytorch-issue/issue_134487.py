# torch.rand(1, 1, 1, 1, dtype=torch.complex128)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, atol=1e-5):
        super().__init__()
        self.atol = atol  # Absolute tolerance for comparison
    
    def forward(self, x):
        # Compute CPU and GPU outputs
        x_cpu = x.to('cpu')
        out_cpu = torch.acos(x_cpu)
        
        # Ensure CUDA is available (as per the issue context)
        x_gpu = x.to('cuda')
        out_gpu = torch.acos(x_gpu)
        
        # Compare outputs after moving GPU result to CPU
        diff = torch.abs(out_cpu - out_gpu.cpu())
        is_close = torch.all(diff < self.atol)
        return torch.tensor(int(is_close.item()))  # 1 if within tolerance, 0 otherwise

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces the input from the issue to trigger CPU/GPU discrepancy
    input_tensor = torch.tensor([(-6.254598811526374e+17 + 0j)], dtype=torch.complex128)
    return input_tensor.view(1, 1, 1, 1)  # Reshape to B=1,C=1,H=1,W=1

