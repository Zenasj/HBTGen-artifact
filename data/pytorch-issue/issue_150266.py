# torch.rand(100, 2048, 512, dtype=torch.float32)  # Inferred input shape

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
    def __init__(self, n_layers, conv_stride):
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

    def forward(self, x):
        return self.model(x)

def my_model_function(n_layers=2, conv_stride=2):
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(n_layers, conv_stride)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((100, 2048, 512), device="cuda")

