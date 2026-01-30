py
from typing import Optional, Tuple

import torch

def fun() -> int:
    # fails
    future: Optional[
        torch.jit.Future[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = None
    # works
    future: Optional[torch.jit.Future[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
    return 1

torch.jit.script(fun)

from typing import Optional, Tuple, List

import torch
from torch.jit import Future

def fun() -> int:
    future: Optional[
        Future[Tuple[torch.Tensor]]
    ] = None
    # works
    future: Optional[torch.jit.Future[Tuple[torch.Tensor]]] = None
    return 1

torch.jit.script(fun)