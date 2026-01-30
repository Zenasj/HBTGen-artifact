import torch.nn as nn

import torch
from torch.autograd import gradgradcheck
from torch.nn.functional import cross_entropy

device = "cpu"

torch.manual_seed(0)
input = torch.randn((1, 2), device=device, dtype=torch.float64).requires_grad_(True)
target = torch.randint(0, 2, (1,), device=device, dtype=torch.int64)
weight = torch.tensor([1.0, -1.0], device=device, dtype=torch.float64)

gradgradcheck(
    lambda input, target: cross_entropy(input, target, weight=weight),
    (input, target),
)

import torch
from torch.autograd import gradgradcheck
from torch.nn.functional import nll_loss, log_softmax

device = "cpu"

torch.manual_seed(0)
input = log_softmax(
    torch.randn((1, 2), device=device, dtype=torch.float64), dim=1
).requires_grad_(True)
target = torch.randint(0, 2, (1,), device=device, dtype=torch.int64)
weight = torch.tensor([1.0, -1.0], device=device, dtype=torch.float64)

gradgradcheck(
    lambda input, target, weight: nll_loss(input, target, weight=weight),
    (input, target, weight),
)