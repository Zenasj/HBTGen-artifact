import torch
import numpy as np


@torch.compile(backend="eager")
def fn(x):
    x = x ** 2
    #torch._dynamo.comptime.graph_break()
    print("HI")
    return 2 * x


fn(np.arange(8))