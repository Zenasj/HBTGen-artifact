import torch

# add_

x = torch.ones(5, 3)
p = torch.ones(5, 3)
t = torch.ones(5, 3)

with fwAD.dual_level():
    dual = fwAD.make_dual(p, t)
    x.add_(dual)  # OK


# index_add_

x = torch.ones(5, 3)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
p = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)

index = torch.tensor([0, 4, 2])

with fwAD.dual_level():
    dual = fwAD.make_dual(p, t)
    x.index_add_(0, index, dual)  # RuntimeError: ZeroTensors are immutable.