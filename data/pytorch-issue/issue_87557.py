# torch.rand(100, 3, 1, 10, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_cpu = nn.InstanceNorm3d(3).to('cpu')
        self.norm_cuda = nn.InstanceNorm3d(3).to('cuda')

    def forward(self, x):
        # Process on CPU
        x_cpu = x.to('cpu', non_blocking=True)
        out_cpu = self.norm_cpu(x_cpu)
        
        # Process on CUDA
        x_cuda = x.to('cuda', non_blocking=True)
        out_cuda = self.norm_cuda(x_cuda)
        
        # Compute maximum difference between outputs
        max_diff = torch.max(torch.abs(out_cpu - out_cuda.to('cpu')))  # Move CUDA result to CPU for comparison
        
        # Return True if difference is within expected floating-point tolerance (1e-7)
        return max_diff <= 1e-7

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, 3, 1, 10, 10, dtype=torch.float32)

