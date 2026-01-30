import torch.nn as nn

import torch
from torch import Tensor
from typing import Tuple
from torch._export import capture_pre_autograd_graph
from torch.export.wrapper import WrapperModule

class FullAttentionState():
    def __init__(self, k: Tensor, v: Tensor, max_seq_len: int) -> None:
        batch_size, num_heads, seq_len, head_dim = k.shape

        self.k = k.new_empty((batch_size, num_heads, max_seq_len, head_dim))
        self.v = v.new_empty((batch_size, num_heads, max_seq_len, head_dim))

        self.k[:, :, :seq_len] = k
        self.v[:, :, :seq_len] = v

        self.seq_len = seq_len


    def append(self, k: Tensor, v: Tensor, constrain: bool = False) -> None:
        
        pos = self.seq_len
        
        self.k[:, :, pos : pos + 1] = k
        self.v[:, :, pos : pos + 1] = v

        if constrain:
            torch._constrain_as_value(self.seq_len, min = 1, max = 1000)

        self.seq_len += 1

    def get(self) -> Tuple[Tensor, Tensor]:
        k = self.k[:, :, : self.seq_len]
        v = self.v[:, :, : self.seq_len]

        return k, v

class SeamlessRepro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

def test(x: Tensor, y: Tensor, k: Tensor, v: Tensor, seq: Tensor) -> Tensor:
    state = FullAttentionState(k, v, k.shape[2])
    state.seq_len = seq.item()
    torch._check_is_size(state.seq_len)
    torch._constrain_as_value(state.seq_len, min = 0, max = 1000)
    state.append(x, y)
    k,v =  state.get()
    # By the time we hit this call seq_len is always >= 1. But how do we
    # let the compiler know that?
    a = torch.matmul(k, v.transpose(-1, -2))
    return a

model = SeamlessRepro()
model.eval()
inputs = (torch.randn(1, 16), torch.randn(1, 16), torch.randn(5, 16, 1002, 16), torch.randn(5, 16, 1002, 16), torch.tensor([0]))
b = torch.export.export(WrapperModule(test), inputs)

def test(x: Tensor, y: Tensor, k: Tensor, v: Tensor, seq: Tensor) -> Tensor:
    state = FullAttentionState(k, v, k.shape[2])
    state.seq_len = seq.item()
    torch._constrain_as_size(state.seq_len, min=0, max = 1000)
    state.append(x, y)
    k,v =  state.get()
    a = torch.matmul(k, v.transpose(-1, -2))
    return a

model = SeamlessRepro()
model.eval()
inputs = (torch.randn(1, 16), torch.randn(1, 16), torch.randn(5, 16, 1002, 16), torch.randn(5, 16, 1002, 16), torch.tensor([0]))
b = torch.export.export(WrapperModule(test), inputs)