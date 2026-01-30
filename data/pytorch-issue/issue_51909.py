import torch.nn as nn

import torch.nn.functional as F
import torch
from torch.fx import symbolic_trace

def f(x):
    return F.normalize(x)

gm = symbolic_trace(f)
torch.jit.script(gm)