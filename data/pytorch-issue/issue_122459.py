# torch.rand(8, 32, dtype=torch.float32, requires_grad=True)
import torch
import torch.nn as nn

# Mock the fully_shard function as it's part of PyTorch's internal FSDP
def fully_shard(module, reshard_after_forward):
    # Placeholder to mimic sharding configuration
    pass

class MyModel(nn.Module):
    def __init__(self, lin_dim, device):
        super().__init__()
        self.in_proj = nn.Linear(lin_dim, lin_dim, device=device)
        self.out_proj = nn.Linear(lin_dim, lin_dim, device=device)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.out_proj(x)
        return x

def my_model_function():
    lin_dim = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(lin_dim, device)
    reshard_after_forward = 2
    fully_shard(model.in_proj, reshard_after_forward=reshard_after_forward)
    fully_shard(model.out_proj, reshard_after_forward=reshard_after_forward)
    fully_shard(model, reshard_after_forward=reshard_after_forward)
    return model

def GetInput():
    return torch.randn((8, 32), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), requires_grad=True)

