# torch.rand(2, 5, 4, dtype=torch.float32)

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List

bs = 2
n_feat = 4
n_time_steps = 5

class Cell(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        return x + state, state

class MyRNN(nn.Module):
    def __init__(self, trace=True):
        super().__init__()
        example_cell_input = torch.rand((bs, n_feat))
        example_cell_state = torch.rand((bs, n_feat))
        self.cell = Cell()
        if trace:
            self.cell = torch.jit.trace(
                self.cell, (example_cell_input, example_cell_state)
            )
            print(self.cell.graph)

    def forward(
        self, x: Tensor, state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if state is None:
            state = x[:, 0]
        inputs = torch.unbind(x, dim=1)
        outputs: List[Tensor] = []
        for input in inputs:
            out, state = self.cell(input, state)
            outputs.append(out)
        return torch.stack(outputs, dim=1), state

class MyModel(nn.Module):
    def __init__(self, trace=True):
        super().__init__()
        self.rnn = MyRNN(trace)

    def forward(self, x):
        return self.rnn(x)

def my_model_function():
    return MyModel(trace=True)

def GetInput():
    return torch.rand(bs, n_time_steps, n_feat, dtype=torch.float32)

