# torch.rand(2, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Indices tensor must match source's shape (2,5) to avoid size mismatch errors
        self.register_buffer('indices', torch.tensor([[1]*5, [2]*5]))  # Shape (2,5)
    
    def forward(self, src):
        # Compute CPU version
        base_cpu = torch.ones(3, 5, device='cpu')
        indices_cpu = self.indices.to(base_cpu.device)
        src_cpu = src.to(base_cpu.device)
        cpu_result = base_cpu.scatter_add_(1, indices_cpu, src_cpu)
        
        # Compute GPU version
        base_gpu = torch.ones(3, 5, device='cuda')
        indices_gpu = self.indices.to(base_gpu.device)
        src_gpu = src.to(base_gpu.device)
        gpu_result = base_gpu.scatter_add_(1, indices_gpu, src_gpu)
        
        # Compare results across devices
        diff = torch.abs(cpu_result - gpu_result.cpu())
        # Return True if non-deterministic difference exceeds 1e-6 threshold
        return torch.max(diff) > 1e-6  # Matches user-reported precision threshold

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 5)

