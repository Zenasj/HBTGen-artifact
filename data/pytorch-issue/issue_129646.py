import torch

from torch import Tensor
reveal_type(Tensor.__contains__)
# mypy:  "def (torch._tensor.Tensor, Any) -> Any"
# pyright: "(self: Tensor, element: Unknown) -> (Any | int | float | bool | Unknown)"