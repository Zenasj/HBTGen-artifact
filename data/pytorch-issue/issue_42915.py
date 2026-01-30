import torch
@torch.jit.script
def aa(a:bool, b:bool):
    a = a | b # Also works &
    return a

import torch
@torch.jit.script
def aa(a:bool, b:bool):
    a |= b # Neither works &
    return a