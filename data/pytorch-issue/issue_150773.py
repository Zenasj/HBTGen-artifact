# torch.rand(100, 2048, 512, dtype=torch.float32)
import torch
from torch import nn

class PermuteModule(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        assert len(x.shape) == len(self.permutation), f"Dimension mismatch! Unable to permute {len(x.shape)} dim input with a {len(self.permutation)} dim permutation!"
        return x.permute(*self.permutation)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(2):  # n_layers=2 as in the test case
            layers.append(PermuteModule((0, 2, 1)))  # (N, L, C) → (N, C, L)
            layers.append(nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=9,
                stride=2,  # Matches conv_stride=2 in the failing test
                padding=0,
                groups=1,
                bias=False
            ))
            layers.append(PermuteModule((0, 2, 1)))  # (N, C, L') → (N, L', C)
            layers.append(nn.LayerNorm(512))  # Operates on channel dimension
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape from the repro script
    return torch.rand(100, 2048, 512, dtype=torch.float32, device="cuda")

