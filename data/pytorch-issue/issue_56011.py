import torch
from typing import List, Optional

@torch.jit.script
def fn():
    x: Optional[List[torch.Tensor]] = [torch.tensor(3)]
    if torch.jit.isinstance(x, List[torch.Tensor]): 
        x.append(torch.tensor(3))
    return x

fn()