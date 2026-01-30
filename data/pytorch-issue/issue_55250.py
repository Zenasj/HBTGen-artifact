import torch
from typing import Tuple

def fn(a: Tuple[int, float], b: Tuple[int]):
    return a + b

a = (1, 2.0)
b = (3,)

print(f"Python: {fn(a, b)}")
scripted = torch.jit.script(fn)
print(f"TorchScript: {scripted(a, b)}")

Python: (1, 2.0, 3)
TorchScript: (1, 2.0, 3)