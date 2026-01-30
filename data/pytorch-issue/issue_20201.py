import torch

a = torch.randn(5)
b = a[a > 0]
b.fill_(0)  # doesn't modify `a`
# BUT
a[a > 0] = 0  # changes `a`

a = torch.rand(5, 5)
b = a[0]
b.fill_(0)
print(a)  # modified!