import torch

@torch.jit.script 
def f(bits): 
    # type: (int) -> int 
    i = 0 
    for _ in range(2**bits): 
         i = i + 1 
    return i

f(8)