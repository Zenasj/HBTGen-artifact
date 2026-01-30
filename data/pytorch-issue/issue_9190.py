import torch
import numpy as np

a = torch.Tensor([1, 2, 3])

# works, correctly produces tensor([-1.,  2.,  3.])
b = torch.where(a == 1, torch.Tensor([-1]), a)

# fails: TypeError: where(): argument 'input' (position 2) must be Tensor, not int
c = torch.where(a == 1, -1, a)

# works in numpy
np.where(a.numpy() == 1, -1, a.numpy())

# both fail: RuntimeError: expected scalar type Long but found Float
d = torch.where(a == 1, torch.tensor(-1), a)
d = torch.where(a == 1, torch.tensor([-1]), a)

torch.where(torch.tensor([False, True]).cuda(), torch.rand(2).cuda(), 0)
# problem with python scalar

torch.where(torch.tensor([False, True]).cuda(), torch.rand(2).cuda(), torch.tensor(0.))
# problem with device

torch.where(torch.tensor([False, True]).cuda(), torch.rand(2).cuda(), torch.tensor(0).cuda())
# problem with dtype

torch.where(torch.tensor([False, True]).cuda(), torch.rand(2).cuda(), torch.tensor(0))
# problem with dtype and device