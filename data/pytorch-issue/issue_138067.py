import torch.nn as nn

from torch import nn, tensor

nn.Linear(1, 1, device="cuda")(tensor([1.], device="cuda"))

from torch import nn, tensor

nn.Linear(1, 1, device="cuda")(tensor([1.], device="cuda"))