# torch.rand(1, 2001, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute on CPU
        x_cpu = x.to('cpu')
        _, idx_cpu = torch.max(x_cpu, 1)
        
        # Compute on GPU if available, else return False (no discrepancy)
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            x_gpu = x.to('cuda:0')
            _, idx_gpu = torch.max(x_gpu, 1)
            idx_gpu = idx_gpu.to('cpu')  # Move to CPU for comparison
        else:
            # If no CUDA, indices will trivially match (no discrepancy)
            idx_gpu = idx_cpu.clone()
        
        # Return True if any indices differ between CPU and GPU
        return torch.ne(idx_cpu, idx_gpu).any()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2001, dtype=torch.float32)

