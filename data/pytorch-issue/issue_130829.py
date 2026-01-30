import torch

t = torch.tensor([1.0, 2.0])

out = tuple(torch.tensor([]) for i in range(2))
torch.unbind_copy(t, out=out)
print('zero', out)

out = tuple(torch.tensor([9.0]) for i in range(2))
torch.unbind_copy(t, out=out)
print('one', out)

out = tuple(torch.tensor([9.0, 99.0]) for i in range(2))
torch.unbind_copy(t, out=out)
print('two', out)

# For two or more dimensions

t = torch.randn(2, 3)

out = tuple(torch.tensor([]) for t in range(2))
torch.unbind_copy(t, out=out)

zero (tensor([1.]), tensor([2.]))
one (tensor([1.]), tensor([2.]))
two (tensor([1.]), tensor([2.]))

import torch

eye = torch.eye(3)
out = torch.zeros(3)

print(torch.select_copy(eye, 0, 0))
print(torch.select_copy(eye, 0, 0, out=out))

out = (torch.zeros(3), torch.zeros(3), torch.zeros(3))
print(torch.unbind_copy(eye))
print(torch.unbind_copy(eye, out=out))

tensor([1., 0., 0.])
tensor([1., 0., 0.])
(tensor([1., 0., 0.]), tensor([0., 1., 0.]), tensor([0., 0., 1.]))
None

tensor([1., 0., 0.])
tensor([1., 0., 0.])
(tensor([1., 0., 0.]), tensor([0., 1., 0.]), tensor([0., 0., 1.]))
(tensor([1., 0., 0.]), tensor([0., 1., 0.]), tensor([0., 0., 1.]))