import torch.nn as nn

py
import torch
from torch import nn
from typing import Final

class Net(nn.Linear):
    x: Final[int]

    def __init__(self):
        super().__init__(5, 10)
        self.x = 0

net = torch.jit.script(Net())