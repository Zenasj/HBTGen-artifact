# torch.rand(32, 800, 192, dtype=torch.float32, device='cuda')  # Add a comment line at the top with the inferred input shape
import torch
from torch import nn
from typing import List

class Permute(nn.Module):
    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        self._permute_idx: List[int] = [from_dims.index(d) for d in to_dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self._permute_idx)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fast_network = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(12, 16)),
            Permute("NTCW", "NCTW"),
            nn.Conv2d(12, 12, kernel_size=(51, 1)),
        )
        self.slow_network = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(16, 12)),
            Permute("NTWC", "NCTW"),
            nn.Conv2d(12, 12, kernel_size=(51, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fast_output = self.fast_network(x)
        slow_output = self.slow_network(x)
        return fast_output, slow_output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(32, 800, 192, dtype=torch.float32, device='cuda')

