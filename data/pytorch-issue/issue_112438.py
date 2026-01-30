from typing import Any, Callable

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.utils import _apply_to_tensors


def hook(*args, **kwargs):
    assert 0, "Hello world!"


def register_backward_hook(output: Any, hook: Callable):
    if not torch.is_grad_enabled():
        return output

    def _register_hook(tensor: torch.Tensor):
        if tensor.requires_grad:
            tensor.register_hook(
                lambda *args, **kwargs: Variable._execution_engine.queue_callback(hook)
            )
        return tensor

    return _apply_to_tensors(_register_hook, output)


model = nn.Sequential(
    nn.Linear(3, 3, device="cuda"), nn.ReLU(), nn.Linear(3, 3, device="cuda")
)
out = model(torch.randn(2, 3, device="cuda"))
register_backward_hook(out, hook)
out.sum().backward()