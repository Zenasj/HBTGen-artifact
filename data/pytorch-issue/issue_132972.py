import torch
from typing import Optional, List

def f(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: Optional[float] = None, causal: bool = False, window_size: Optional[List[int]] = None, softcap: float = 0.0, alibi_slopes: Optional[List[float]] = None, block_table: Optional[torch.Tensor] = None) -> torch.Tensor:
    return k + 1 + v + q

f = torch.library.custom_op("xxx::f", f, mutates_args="unknown")