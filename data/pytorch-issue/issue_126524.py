import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self, cache: torch.Tensor):
        super().__init__()
        assert cache.ndim == 3
        self.cache = torch.nn.Parameter(cache, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = x.size(1)
        self.cache = torch.roll(self.cache, -n_tokens, dims=1)
        self.cache[:, -n_tokens:, :] = x
        return self.cache

MyModule(torch.zeros(2, 3, 4))(torch.zeros(2, 2, 4))