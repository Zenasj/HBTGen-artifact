import torch

some_variable = 1e-310
some_tensor = torch.rand((5,5))

def func(x: torch.Tensor):
    return x**2

print("Before compile:", some_variable)
func_compiled = torch.compile(func)
func_compiled(some_tensor)
print("After compile:", some_variable)