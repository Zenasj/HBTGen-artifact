import torch.nn as nn

import torch
from torch import nn
from tqdm.auto import tqdm

X = torch.rand(32, 800, 192).to("cuda")

class Permute(nn.Module):
    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        self._permute_idx: List[int] = [from_dims.index(d) for d in to_dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)
        
### FAST ORDER (81 iterations / second)
network = nn.Sequential(
    nn.Unflatten(dim=-1, unflattened_size=(12, 16)),
    Permute("NTCW", "NCTW"),
    nn.Conv2d(12, 12, kernel_size=(51, 1)),
).to("cuda")

torch.cuda.empty_cache()
for __ in tqdm(range(100)):
    out = network.forward(X)
    torch.cuda.synchronize()

### Slow Order (3.79 iterations / second)
network = nn.Sequential(
    nn.Unflatten(dim=-1, unflattened_size=(16, 12)),
    Permute("NTWC", "NCTW"),
    nn.Conv2d(12, 12, kernel_size=(51, 1)),
).to("cuda")
torch.cuda.empty_cache()
for __ in tqdm(range(100)):
    out = network.forward(X)
    torch.cuda.synchronize()