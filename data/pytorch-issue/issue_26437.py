# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torch import Tensor
from collections import namedtuple

_GoogLeNetOutputs = namedtuple('_GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.identity = nn.Identity()  # Dummy layer to mimic model structure

    def forward(self, x: Tensor) -> _GoogLeNetOutputs:
        # type: (Tensor) -> _GoogLeNetOutputs
        # Simulate output structure with named tuple (logits, aux_logits2, aux_logits1)
        return _GoogLeNetOutputs(x, x, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

