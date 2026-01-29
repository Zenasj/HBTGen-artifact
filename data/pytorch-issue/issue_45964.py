# torch.rand(8, 17, dtype=torch.float32, device='cuda')  # Inferred input shape (T=8, B=17 to trigger non-determinism)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        T, B = x.shape
        # Generate indices based on input shape
        ts = torch.arange(T, device=x.device)
        bs = torch.arange(B, device=x.device)
        ts, bs = torch.meshgrid(ts, bs, indexing='ij')  # Create (T,B) meshgrid with 'ij' indexing
        ts_inv = torch.flip(ts, (0,))  # Flip rows of ts to create inverted indices
        
        # Create a copy to modify and trigger in-place assignment behavior
        result = x.clone()
        # Perform the problematic assignment: writes to result[ts_inv, bs] using original x values
        result[ts_inv, bs] = x  
        return result

def my_model_function():
    return MyModel()

def GetInput():
    # Use B=17 to trigger non-deterministic behavior as described in the issue
    return torch.rand(8, 17, device='cuda', dtype=torch.float32)

