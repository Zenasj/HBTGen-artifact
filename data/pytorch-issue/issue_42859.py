from typing import Optional, List
import torch
@torch.jit.script
def buggy_fun(x:torch.Tensor, classes:Optional[List]=None):
    
    # Filter by class
    if classes:
        x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]