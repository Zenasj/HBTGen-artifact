import torch
import contextlib

@contextlib.contextmanager
def set_default_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)

@torch.compile(backend="eager", fullgraph=True)
def fn():
    with set_default_dtype(torch.float64):
        x = torch.tensor([3.0, 3.0 + 5.0j])
    return x