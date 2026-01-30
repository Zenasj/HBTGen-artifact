from __future__ import annotations

import torch

@torch.library.custom_op("mylibrary::foo_op", mutates_args={})
def foo_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()