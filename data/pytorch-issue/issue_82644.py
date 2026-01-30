import torch

t = torch.tensor([0.4901], requires_grad=True)
# Works
t.prod(0).backward()

t = torch.tensor([0.], requires_grad=True)
# RuntimeError: Function ProdBackward1 returned an invalid gradient at
# index 0 - got [] but expected shape compatible with [1]
t.prod(0).backward()