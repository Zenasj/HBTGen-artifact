python
# file mypy_test.py
from typing import cast, Any, TYPE_CHECKING

import torch


class Tensor(torch.Tensor):
    """Type wrapper for torch.Tensor"""
    def __new__(cls, *args: Any, **kwargs: Any) -> "Tensor":
        return cast(Tensor, torch.as_tensor(*args, **kwargs))


GLOBAL: Tensor = Tensor([0, 0], dtype=torch.float32)


class Foo:
    a: Tensor = GLOBAL

    def foo(self) -> Tensor:
        return cast(Tensor, self.a)

if TYPE_CHECKING:
    reveal_type(GLOBAL)
    reveal_type(Foo.a)

if __name__ == "__main__":
    # this works
    bar: Tensor = GLOBAL
    bar = Foo().foo()
    # this doesn't work in 1.8 but does in 1.7
    bar = Foo().a