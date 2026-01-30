import torch
import numpy as np

def func():
    x = np.array([1, 2, 3])
    return np.array([np.sin(x), np.cos(x)])

print(func())
o_func = torch.compile(fullgraph=True)(func)
print(o_func())