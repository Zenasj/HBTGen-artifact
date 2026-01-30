import torch.nn as nn

import torch

class Cache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key_cache = []

    def __len__(self):
        return len(self.key_cache)

@torch.compile(backend="eager")
def f(x):
    cache = Cache()
    if cache:
        return x + 1
    return x + 2

f(torch.ones(1))