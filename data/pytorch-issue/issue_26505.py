import torch

@torch.jit.script
def f(y):
    # type: (Tuple[str, str, str, str])
    return list(y)