import torch.nn as nn

from typing import Any, Dict, Tuple

import torch
from torch import nn


def forward_pre_hook(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
) -> None:
    ...


def forward_pre_hook_return_input(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    ...


def forward_pre_hook_with_kwargs(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
    kwargs: Dict[str, Any],
) -> None:
    ...


def forward_pre_hook_with_kwargs_return_input(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]:
    ...


def forward_hook(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> None:
    ...


def forward_hook_return_output(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    ...


def forward_hook_with_kwargs(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
    kwargs: Dict[str, Any],
    output: torch.Tensor,
) -> None:
    ...


def forward_hook_with_kwargs_return_output(
    module: nn.Linear,
    args: Tuple[torch.Tensor, ...],
    kwargs: Dict[str, Any],
    output: torch.Tensor,
) -> torch.Tensor:
    ...

model = nn.Module()

# OK
model.register_forward_pre_hook(forward_pre_hook)
model.register_forward_pre_hook(forward_pre_hook_return_input)
model.register_forward_pre_hook(forward_pre_hook_with_kwargs, with_kwargs=True)
model.register_forward_pre_hook(forward_pre_hook_with_kwargs_return_input, with_kwargs=True)

model.register_forward_hook(forward_hook)
model.register_forward_hook(forward_hook_return_output)
model.register_forward_hook(forward_hook_with_kwargs, with_kwargs=True)
model.register_forward_hook(forward_hook_with_kwargs_return_output, with_kwargs=True)

# mypy(error): [arg-type]
model.register_forward_pre_hook(forward_hook)
model.register_forward_pre_hook(forward_hook_return_output)
model.register_forward_pre_hook(forward_hook_with_kwargs)
model.register_forward_pre_hook(forward_hook_with_kwargs_return_output)

model.register_forward_hook(forward_pre_hook)
model.register_forward_hook(forward_pre_hook_return_input)

# false negatives
model.register_forward_hook(forward_pre_hook_with_kwargs)
model.register_forward_hook(forward_pre_hook_with_kwargs_return_input)

model.register_forward_pre_hook(forward_pre_hook_with_kwargs, with_kwargs=False)
model.register_forward_pre_hook(forward_pre_hook_with_kwargs_return_input, with_kwargs=False)
...