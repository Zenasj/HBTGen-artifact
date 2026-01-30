import torch.nn as nn

import torch
import torch.nn.functional as F

# works
shape = (3, 5)
input = torch.randn(*shape).log_softmax(dim=-1)
target = torch.randn(*shape).softmax(dim=-1)
F.cross_entropy(input, target)

# crashes
shape = (3,)
input = torch.randn(*shape).log_softmax(dim=-1)
target = torch.randn(*shape).softmax(dim=-1)
F.cross_entropy(input, target)
# IndexError:  Dimension out of range(expected to be in range of [-1, 0], but got 1)