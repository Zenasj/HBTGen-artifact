import torch

py
from typing import List

def f(list_or_int: List[int]):
    if isinstance(list_or_int, int):
        the_int = list_or_int
    else:
        the_int = list_or_int[0]
        
def g(list_or_int: List[int]):
    the_int = list_or_int if isinstance(list_or_int, int) else list_or_int[0]
    
torch.jit.script(f)  # OK
torch.jit.script(g)  # not OK