import torch
from torch import nn

def sliding_window(b, h, q_idx, kv_idx, val):
    return (q_idx - kv_idx).abs() < val

# torch.randint(0, 1024, (2, 1024), dtype=torch.long)  # Input is a 2x1024 tensor of indices
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.val = nn.Parameter(torch.randn(()))  # Parameter replacing partial's tensor
        
    def forward(self, indices):
        q_idx, kv_idx = indices[0], indices[1]
        # Define mask function using model's parameter
        def mask_mod(b, h, q_i, kv_i):
            return sliding_window(b, h, q_i, kv_i, self.val)
        
        # Simulate create_block_mask call with fixed sequence lengths
        # The actual implementation depends on flex_attention's requirements
        # Here we return a dummy mask for demonstration
        return torch.ones(1024, 1024, dtype=torch.bool)  # Replace with actual mask creation

def my_model_function():
    return MyModel()

def GetInput():
    # Generate indices tensor (q_idx, kv_idx) with shape (2, 1024)
    return torch.randint(0, 1024, (2, 1024), dtype=torch.long, device='cuda')

