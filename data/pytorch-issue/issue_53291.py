import math

import torch
import cmath

def fn(a: complex):
        return cmath.log(a)

scripted = torch.jit.script(fn)
print(scripted(3 + 5j))