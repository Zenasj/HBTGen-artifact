import torch

@torch.jit.script
def foo(parameter_tensor):
    size = 0
    size += parameter_tensor.numel() * parameter_tensor.element_size()
    return size