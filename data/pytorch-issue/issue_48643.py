import torch.nn as nn

from torch import nn, jit, Tensor
from typing import List


class Foo(nn.Module):
    @jit._overload_method
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @jit._overload_method  # type: ignore[no-redef]
    def forward(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[no-redef]
        pass


foo = Foo()
jit.script(foo)

from torch import nn, jit, Tensor


class Bar(nn.ModuleDict):
    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        pass


bar = Bar()
jit.script(bar)