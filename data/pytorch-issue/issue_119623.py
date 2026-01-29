# Inputs: (torch.rand(1, 16), torch.rand(1, 16), torch.rand(5, 16, 1002, 16), torch.rand(5, 16, 1002, 16), torch.tensor([0]))
import torch
from torch import Tensor
from typing import Tuple

class MyModel(torch.nn.Module):
    class FullAttentionState:
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
                torch._constrain_as_size(self.seq_len, min=1, max=1000)
            self.seq_len += 1

        def get(self) -> Tuple[Tensor, Tensor]:
            k = self.k[:, :, : self.seq_len]
            v = self.v[:, :, : self.seq_len]
            return k, v

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        x, y, k, v, seq = inputs
        state = self.FullAttentionState(k, v, k.shape[2])
        state.seq_len = seq.item()
        torch._constrain_as_size(state.seq_len, min=0, max=1000)
        state.append(x, y)
        k, v = state.get()
        a = torch.matmul(k, v.transpose(-1, -2))
        return a

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(1, 16)
    y = torch.rand(1, 16)
    k = torch.rand(5, 16, 1002, 16)
    v = torch.rand(5, 16, 1002, 16)
    seq = torch.tensor([0])
    return (x, y, k, v, seq)

