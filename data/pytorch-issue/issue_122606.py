import torch.nn as nn

import torch

@torch.compile
def cast_and_pad(x):
    return torch.nn.functional.pad(x.to(torch.float32), (0, 3, 0, 0))

x=torch.ones(1, 1, 13, dtype=torch.int64)
print(cast_and_pad(x))