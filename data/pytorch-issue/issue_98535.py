import torch
import torch._dynamo as dynamo
import logging
dynamo.config.set_loggers_level(logging.DEBUG)


def f(idx: int, y: torch.Tensor, z: torch.Tensor):
    if idx in z:
        y[idx] += 1
    return y


f = dynamo.optimize('eager', dynamic=True)(f)

f(10, torch.tensor([0, 0, 0]), torch.tensor([1, 2, 10]))