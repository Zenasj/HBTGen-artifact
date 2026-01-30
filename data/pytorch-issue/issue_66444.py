import torch.nn as nn

from contextlib import contextmanager
import torch

from torch._C import _DisableTorchDispatch
from torch.utils._python_dispatch import enable_python_mode


n = torch.ones([10, 10])  

torch.save(n, "python-mode")


@contextmanager
def no_dispatch():
    guard = _DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


class TestTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        with no_dispatch():
            return func(*args, **kwargs)


with enable_python_mode(TestTensor):
    m = torch.load("python-mode")

py
import torch
from copy import deepcopy

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # this works
        default = torch.nn.parameter.Parameter(torch.tensor(1.0))
        deepcopy(default)

        # this doesn't
        default = torch.tensor(1.0)
        deepcopy(default)

torch.distributed.nn.utils.init_meta(TestModule)

"""
  deepcopy(default)
  File "/Users/thomas/.pyenv/versions/3.8.5/lib/python3.8/copy.py", line 153, in deepcopy
    y = copier(memo)
  File "/Users/thomas/Documents/GitHub/pytorch-lightning/.venv/lib/python3.8/site-packages/torch/_tensor.py", line 120, in __deepcopy__
    new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
RuntimeError: Missing cases in 'toPyObject'! Can't convert Storage to a Python object
"""