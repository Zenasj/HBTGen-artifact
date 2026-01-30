from typing import Dict, Optional
from torch import Tensor

import torch


def cholesky(M: Tensor, diagnostics: Optional[Dict[str, Tensor]] = None) -> Tensor:
    L, info = torch.linalg.cholesky_ex(M)
    if diagnostics is not None:
        diagnostics["info"] = info
    return L