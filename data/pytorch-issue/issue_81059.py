import torch
from torch.testing._internal.composite_compliance import check_backward_formula

x = torch.tensor([[1., 1.], [1., 0.]], requires_grad=True)
args = (x, 1)

check_backward_formula_callable(torch.prod, args, {})