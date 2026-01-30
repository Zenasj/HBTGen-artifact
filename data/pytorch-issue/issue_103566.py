import torch.nn as nn

import torch
model = torch.nn.Linear(10,20)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

_use_grad.__doc__ = func.__doc__
_use_grad.__signature__ = inspect.signature(func)

def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    _use_grad.__doc__ = func.__doc__
    _use_grad.__signature__ = inspect.signature(func)
    return _use_grad