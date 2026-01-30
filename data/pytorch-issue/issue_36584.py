import torch.nn as nn

# Bring datatype enums to the main namespace
class DataType:
    pass

def _InitDataType():
    for name, value in caffe2_pb2.TensorProto.DataType.items():
        setattr(DataType, name, value)

_InitDataType()

from functools import wraps
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


PREDICT_BATCH_SIZE = 1024


def uniform_weight_initializer(module: nn.Module, lower: float, upper: float) -> None:
    """
    Initialize weights from Unif(lower, upper). Initialize biases as zero.
    Initialization is performed in-place.
    """
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, lower, upper)
        if module.bias is not None:
            nn.init.zeros_(module.bias) # ERROR HERE

if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, lower, upper)
        if module.bias is not None:
            nn.init.zeros_(module.bias) # ERROR HERE

import torch
from torch import Tensor, nn


def func(module: nn.Module) -> Tensor:
    res = torch.tensor([])
    #if isinstance(module, nn.Linear):
    nn.init.uniform_(module.weight, 0, 1)
    #if module.bias is not None:
    res = nn.init.zeros_(module.bias)

    return res


module = nn.Linear(10, 30, bias=False)
t1 = torch.randn(256, 10)
t2 = module(t1)
res = func(module)

if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, 0, 1)
        #if module.bias is not None:
        res = nn.init.zeros_(module.bias)

if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, 0, 1)
        reveal_type(module)
        reveal_type(module.bias)
        if module.bias is not None:
            res = nn.init.zeros_(module.bias)

nn.init.zeros_(module.bias)  # pyre-ignore[6]

if module.bias is not None:
            bias = module.bias
            res = nn.init.zeros_(bias)