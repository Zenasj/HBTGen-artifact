# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Generate CPU and CUDA arange tensors with step=0.01 (the problematic case)
        cpu_tensor = torch.arange(-4.0, 4.0, 0.01, dtype=torch.float32, device='cpu')
        cuda_tensor = torch.arange(-4.0, 4.0, 0.01, dtype=torch.float32, device='cuda')
        
        # Compare shapes first (shape mismatch is a critical failure)
        if cpu_tensor.shape != cuda_tensor.shape:
            return torch.tensor(1.0)  # 1.0 indicates shape mismatch
        
        # Compute maximum absolute difference between elements
        # Move CUDA tensor to CPU for comparison
        element_diff = torch.max(torch.abs(cpu_tensor - cuda_tensor.cpu()))
        
        # Return maximum difference (non-zero indicates numerical discrepancy)
        return element_diff

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input (not used by the model, but required for interface compatibility)
    return torch.rand(1, dtype=torch.float32)

