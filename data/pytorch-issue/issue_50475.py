# torch.rand(B, H, W, C_in, dtype=torch.float32)  # e.g., (64*1024, 7, 7, 2)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        layers = [nn.Conv2d(2, channels, 3, 1, 1)]
        for _ in range(16):
            layers.append(nn.Conv2d(channels, channels, 3, 1, 1))
        layers.append(nn.Conv2d(channels, 1, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, obs):
        x = obs.permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.flatten(1).log_softmax(-1)

def my_model_function():
    return MyModel()

def GetInput():
    batch = 64 * 1024
    width = 7
    return torch.rand((batch, width, width, 2), dtype=torch.float32, device='cuda')

