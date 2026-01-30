import torch.nn as nn

python
import torch

bn = torch.nn.BatchNorm1d(num_features=3)

print(bn.weight)  # Parameter containing: tensor([1., 1., 1.], requires_grad=True)
print(bn.bias)    # Parameter containing: tensor([0., 0., 0.], requires_grad=True)

bn = torch.nn.BatchNorm1d(num_features=3, affine=True, use_scale=False)

print(bn.weight)  # None
print(bn.bias)    # Parameter containing: tensor([0., 0., 0.], requires_grad=True)

bn = torch.nn.BatchNorm1d(num_features=3, affine=False, use_scale=False)

print(bn.weight)  # None
print(bn.bias)    # None

m = torch.nn.Sequential(
    torch.nn.BatchNorm1d(3, use_scale=False)
)

m.eval()

x = torch.randn(5, 3)

y1 = m.forward(x)

y1_test = (x-m._modules['0'].running_mean)/torch.sqrt(m._modules['0'].running_var+m._modules['0'].eps) + \
    m._modules['0'].bias

assert torch.all(y1 == y1_test)  # True