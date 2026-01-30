import torch
import numpy as np

def __setitem__(self, indices, values):
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
    super(type(self), self).__setitem__(indices, values)

def __setitem__(self, indices, values):
    if not isinstance(values, torch._TensorBase) and not np.isscalar(values):
        values = type(self)(values)
    super(type(self), self).__setitem__(indices, values)

def __setitem__(self, indices, values):
    ...
    def iterable(obj):
        try:
            iter(obj)
        except TypeError:
            return False
        else:
            return True

    if (isinstance(values, torch._TensorBase) and
            (indices in (slice(None), Ellipsis)) or
            (iterable(indices) and all(i in (slice(None), Ellipsis)
                                       for i in indices))):
        # Covers `x[:] = values`, `x[:, :] = values` etc, and `x[...] = values`
        self.copy_(values)