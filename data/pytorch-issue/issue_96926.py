# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='batch')

    def forward(self, x):
        x = self.conv(x)
        return self.stochastic_depth(x)

class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input

        survival_rate = 1.0 - self.p
        if self.mode == "row":
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
        noise = torch.empty(size, dtype=input.dtype, device=input.device)
        noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return input * noise

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

