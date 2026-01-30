# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Callable, Any

import torch

HANDLED_FUNCTIONS = {}

class MyClass:
    def __init__(self, foo):
        self.foo = foo

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:

        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, MyClass)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self):
        return "MyClass({})".format(self.foo)

def implements_for(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDict."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_for(torch.stack)
def _stack(input, dim=0, *, out=None):
    return MyClass(sum([x.foo for x in input]))

v0 = MyClass(1)
v1 = MyClass(1)
print(torch.stack([v0, v1]))

@torch.compile(fullgraph=True)
def func(v0, v1):
    return torch.stack([v0, v1])

print(func(v0, v1))