from __future__ import annotations

import torch
from typing import List

@torch.library.custom_op("test::custom_op_list", mutates_args=())
def my_custom_op(
    x: torch.Tensor,
) -> List[torch.Tensor]:
    return [torch.randn_like(x)]

from __future__ import annotations

import torch
from typing import List

MyList = List

@torch.library.custom_op("test::custom_op_list", mutates_args=())
def my_custom_op(
    x: torch.Tensor,
) -> MyList[torch.Tensor]:
    return [torch.randn_like(x)]

from __future__ import annotations

import torch

def outer_fn():
    from typing import List
    @torch.library.custom_op("test::custom_op_list", mutates_args=())
    def my_custom_op(
        x: torch.Tensor,
    ) -> MyList[torch.Tensor]:
        return [torch.randn_like(x)]