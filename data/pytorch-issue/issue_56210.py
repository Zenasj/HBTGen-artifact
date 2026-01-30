py
import torch

def fun(x: torch.Tensor) -> torch.Tensor:
    xs: list[torch.Tensor] = []
    xs.append(x)
    return xs.pop()

torch.jit.script(fun)