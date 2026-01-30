import torch

x = torch.randn(...)
x.register_hook(my_hook)
compiled_fn(x)  # <-- could x be cached?