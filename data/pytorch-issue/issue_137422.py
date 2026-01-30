import torch
x = torch.randn(3, 1, 2)
x.squeeze(dim=None)  # ❌ RuntimeError
torch.squeeze(x, dim=None)  # ❌ RuntimeError

from typing import Any, Protocol
from torch import Tensor


class TensorMethodWithDim(Protocol):
    """Callback Protocol for tensor methods supporting `dim` argument.

    NOTE: Adding `*args: Any` and `**kwargs: Any` means we are specifying
      a partial signature for the method.
      see: https://typing.readthedocs.io/en/latest/spec/callables.html#meaning-of-in-callable
    """

    def __call__(
        _,
        self: Tensor,
        dim: None | int,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor: ...


METHODS_WITH_DIM: list[TensorMethodWithDim] = [
    Tensor.mean,     # mypy ✅
    Tensor.sum,      # mypy ✅
    Tensor.std,      # mypy ✅
    Tensor.var,      # mypy ✅
    Tensor.squeeze,  # mypy ❌: List item 4 has incompatible type overloaded function
    Tensor.argmax,   # mypy ✅
    Tensor.argmin,   # mypy ✅
]