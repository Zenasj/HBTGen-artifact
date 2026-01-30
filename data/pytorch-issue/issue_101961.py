import torch.nn as nn

import torch
from torch import nn


t = torch.tensor([1, 2, 3], dtype=torch.float32)

t2 = torch.Tensor._make_subclass(  # OK
    nn.Parameter,
    t.data,
)
reveal_type(t2)  # Type of "t2" is "Parameter"

t3 = t._make_subclass(  # OK
    nn.Parameter,
    t.data,
)
reveal_type(t3)  # Type of "t3" is "Parameter"