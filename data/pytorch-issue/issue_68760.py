import torch.nn as nn

py
import torch
from torch.nn.functional import prelu

torch.manual_seed(1234)

shape = (5, 5)
input = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
weight = torch.randn(5, dtype=torch.float64, requires_grad=True)

torch.autograd.gradgradcheck(prelu, (input, weight), gen_non_contig_grad_outputs=False)  # passes

torch.autograd.gradgradcheck(prelu, (input, weight), gen_non_contig_grad_outputs=True)  # fails