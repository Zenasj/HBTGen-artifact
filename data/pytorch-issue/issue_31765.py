import torch.nn as nn

T_co = TypeVar('T_co', covariant=True)


class Module(Generic[T_co]):
    def __init__(self) -> None: ...

    def forward(self, *input: Any, **kwargs: Any) -> T_co: ...  # type: ignore

    def __call__(self, *input: Any, **kwargs: Any) -> T_co: ...  # type: ignore

import torch
from torch import nn


class SomeModel(nn.Module[torch.Tensor]):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor