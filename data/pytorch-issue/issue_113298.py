import torch

assert torch.is_grad_enabled()

@torch.set_grad_enabled(False)  # unexpectedly, this mutates the grad mode!
def inner_func(x):
    return x.sin()

assert torch.is_grad_enabled()  # AssertionError