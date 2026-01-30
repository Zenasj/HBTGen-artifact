import torch

def remove_nans(tensor):
    if torch._C._get_tracing_state() is None:
        tensor.masked_fill_(...)
    else:
        ... # some equivalent code that the JIT can understand

def remove_nans_masked_fill(tensor):
    tensor.masked_fill_(...)

def remove_nans_tracing(tensor):
    ...

remove_nans = remove_nans_masked_fill