import torch
from einops import rearrange

# torch.rand(3, 3, 32, 32, dtype=torch.float32).cuda()  # Input shape (B, C, H, W)
class MyModel(torch.nn.Module):
    def forward(self, x):
        with torch.autocast(device_type="cuda"):
            return rearrange(x, "B C H W -> B H W C")

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 3, 32, 32, device="cuda")

