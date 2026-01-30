import torch
from torch.autograd.functional import vjp


def exp_reducer(x):
    return x.exp().sum(dim=1)


inputs = torch.rand(4, 4)
v = torch.ones(4)

with torch.no_grad():
    print(vjp(exp_reducer, inputs, v))

with torch.enable_grad():
    print(vjp(exp_reducer, inputs, v))

(tensor([6.9459, 7.6220, 6.1336, 6.0577]), tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]))
(tensor([6.9459, 7.6220, 6.1336, 6.0577]), tensor([[2.2766, 2.3207, 1.3200, 1.0287],
        [2.7153, 1.4122, 1.6791, 1.8154],
        [1.5301, 1.7467, 1.7776, 1.0792],
        [1.2966, 1.0290, 1.8711, 1.8610]]))