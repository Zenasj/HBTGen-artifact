# torch.rand(1, dtype=torch.float32, requires_grad=True)  # Input shape inferred from user's all_gather example
import torch
import torch.distributed as dist
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)  # Dummy layer to preserve gradients (required for grad_fn)
        
    def forward(self, x):
        # Simulate distributed all_gather with gradient-aware setup
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gather_list = [torch.ones_like(x, requires_grad=True) for _ in range(world_size)]
            dist.all_gather(gather_list, x)
            return torch.cat(gather_list)  # Combine for gradient flow demonstration
        return x  # Fallback if not in distributed mode

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32, requires_grad=True)

