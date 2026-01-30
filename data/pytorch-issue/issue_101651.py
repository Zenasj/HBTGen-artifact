import torch

if is_compiled_module(model):
    torch.save(model._orig_mod)