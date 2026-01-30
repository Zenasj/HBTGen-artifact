import torch
from typing import List


@torch.jit.script
def indexing(array: List[int], idx):
    return array[idx]


print(indexing([1, 2, 3], 0))