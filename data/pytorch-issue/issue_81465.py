python
import torch
from typing import List


@torch.jit.script
class Mother():
    def __init__(self):
        pass
    
    
@torch.jit.script
def f():
    l : List[Mother] = []
    for i in range(3):
        obj = Mother()
        l.append(obj)


f()