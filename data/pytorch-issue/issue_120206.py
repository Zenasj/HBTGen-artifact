import torch
from torch.testing import make_tensor
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
t = torch.sparse_coo_tensor(((0, 1), (1, 0)), (1, 2), size=(2, 2))
t2 = FakeTensor.from_tensor(t, FakeTensorMode())
print(repr(t2))