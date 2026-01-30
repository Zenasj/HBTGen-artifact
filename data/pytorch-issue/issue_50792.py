import torch
foo = torch.empty(5).requires_grad_()
foo.count_nonzero()