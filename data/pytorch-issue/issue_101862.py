import torch.nn as nn

class _TensorBase(metaclass=_TensorMeta):
    ...
    def _make_subclass(cls, data: Tensor, require_grad: _bool = False, dispatch_strides: _bool=False, dispatch_device: _bool=False, device_for_backend_keys: Optional[_device] = None) -> Tensor: ...

import torch
from torch import nn


t = torch.tensor([1, 2, 3], dtype=torch.float32)

t2 = torch.Tensor._make_subclass(
    nn.Parameter,  # Argument of type "Type[Parameter]" cannot be assigned to parameter "cls" of type "Tensor" in function "_make_subclass"
    t.data,
)
t3 = t._make_subclass(
    nn.Parameter,  # Argument of type "Type[Parameter]" cannot be assigned to parameter "data" of type "Tensor" in function "_make_subclass"
    t.data,  # Argument of type "Tensor" cannot be assigned to parameter "require_grad" of type "_bool" in function "_make_subclass"
)

S = TypeVar("S", bound="torch.Tensor")

...

class _TensorBase(metaclass=_TensorMeta):
    ...
    @staticmethod
    def _make_subclass(cls: Type[S], data: Tensor, require_grad: _bool = False, dispatch_strides: _bool=False, dispatch_device: _bool=False, device_for_backend_keys: Optional[_device] = None) -> S: ...