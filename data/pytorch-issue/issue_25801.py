import torch

@torch.jit.script
def fn(x):
    # type: (List[int]) -> bool
    return max(x)