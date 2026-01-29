# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (B, C, H, W) where B=100, C=512, H=2048, W=1

import torch
import torch.nn as nn

class PermuteModule(nn.Module):
    def __init__(self, permutation):
        super(PermuteModule, self).__init__()
        self.permutation = permutation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == len(self.permutation), f"Dimension mismatch! Unable to permute {len(x.shape)} dim input with a {len(self.permutation)} dim permutation!"
        return x.permute(*self.permutation)

class MyModel(nn.Module):
    def __init__(self, n_layers: int, conv_stride: int):
        super(MyModel, self).__init__()
        _sequence = []
        for _ in range(n_layers):
            _sequence += [
                PermuteModule((0, 2, 1)),
                nn.Conv1d(in_channels=512, out_channels=512, groups=1, kernel_size=9, dilation=1, stride=conv_stride, padding=0, bias=False),
                PermuteModule((0, 2, 1)),
                nn.LayerNorm(512),
                nn.ReLU()
            ]
        self.model = nn.Sequential(*_sequence).to(device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(n_layers=2, conv_stride=2)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((100, 2048, 512), device="cuda")

