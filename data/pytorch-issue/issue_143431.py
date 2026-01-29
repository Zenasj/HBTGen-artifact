# torch.rand(2, 12, 16, 32, 32, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, image_latent):
        B = image_latent.size(0)
        x = torch.rand(B, 12)
        rand_values = torch.rand(*x.shape)
        indices = torch.argsort(rand_values, dim=-1)[:, :3 + 3]
        selected = image_latent[torch.arange(B).unsqueeze(-1), indices]
        return selected[:, :3]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 12, 16, 32, 32, dtype=torch.float32, device='cuda')

