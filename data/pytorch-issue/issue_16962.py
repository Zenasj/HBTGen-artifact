import torch

@torch.jit.script
def make_list(x):
    # type: (Tuple[int, int, int]) -> List[int]
    return x

@torch.jit.script
def make_list(x):
    # type: (List[int]) -> List[int]
    return x