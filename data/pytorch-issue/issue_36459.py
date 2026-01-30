model = torch.nn.Sequential(*[
    torch.nn.Linear(in_features=200,out_features=100),
    torch.nn.Parallel(*[
        torch.nn.Linear(in_features=100, out_features=16),
        torch.nn.Linear(in_features=100, out_features=16),
    ]),
    torch.nn.Bilinear(in1_features=16,in2_features=16,out_features=100),
])
model(torch.randn(200,))

# Parallel
def forward(self, input):
        output = []
        for module in self:
            output.append(module(input))
        return tuple(output)

# Sequential
def forward(self, input):
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

from typing import Callable, Union

import torch
import torch.nn as nn


class Parallel(nn.ModuleList):
    """Runs modules in parallel on the same input and merges their results."""

    def __init__(self, *modules: nn.Module, merge: Union[str, Callable] = "sum"):
        """Runs modules in parallel on the same input and merges their results.

        Args:
            merge: operation for merging list of results (default: `"sum"`)
        """
        super().__init__(modules)
        self.merge = create_merge(merge)

    def forward(self, x: Tensor) -> Tensor:
        return self.merge([module(x) for module in self])


MERGE_METHODS: Dict[str, Callable] = {
    "cat": lambda xs: torch.cat(xs, dim=1),
    "sum": lambda xs: sum(xs),  # type: ignore
    "prod": lambda xs: reduce(lambda x, y: x * y, xs),  # type: ignore
}


def create_merge(merge: Union[str, Callable]) -> Callable:
    return MERGE_METHODS[merge] if isinstance(merge, str) else merge