import torch

def func(a, b):
    c = (a > 1) & (b > 1) 
    return c

a = torch.ones((10), dtype=torch.int64)
b = torch.ones((10), dtype=torch.uint8)
func_compiled = torch.compile(func)
result = func_compiled(a, b)