import torch.nn as nn

import torch
from torch import Tensor, nn


class BufferList(nn.Module):
    """Like torch.nn.ParameterList, but for buffers instead of parameters"""

    def __init__(self, *buffers: Tensor | None) -> None:
        super().__init__()
        for x in buffers:
            self.register_buffer(str(len(self._buffers)), x)

    def __getitem__(self, idx: int) -> Tensor | None:
        return self._buffers[str(idx + len(self._buffers) if idx < 0 else idx)]

    def __len__(self) -> int:
        return len(self._buffers)


class Dummy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.buflist = BufferList(torch.arange(5.0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.buflist[0]

if __name__ == "__main__":
    x = torch.ones(5)
    model = torch.compile(Dummy())
    print(model(x))