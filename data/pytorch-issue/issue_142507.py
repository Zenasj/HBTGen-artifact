import torch.nn as nn

import torch
from torch import Tensor, nn

b = torch.tensor([0, 2, 4, 6, 8])

@torch.compile
class MyModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.isin(x, b)

model = MyModel()

for i in range(1, 20):
    a = torch.arange(i)
    print(model(a))

import torch
from torch import Tensor

b = torch.tensor([0, 2, 4, 6, 8])

@torch.compile
def forward(x: Tensor) -> Tensor:
    return torch.isin(x, b)

for i in range(1, 20):
    a = torch.arange(i)
    print(forward(a))