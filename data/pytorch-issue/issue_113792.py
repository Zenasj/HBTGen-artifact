import torch
import warnings

def f(x):
    warnings.warn("moo")
    return x + x

ep = torch.export.export(f, (torch.ones(1, 3),))

prints_warning_as_no_op=False