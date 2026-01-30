import torch

@torch.jit.script
def fn(x):
    # type: (Tuple[int, int]) -> int
    return x[0] + x[1]

fn.save('fn.pt')