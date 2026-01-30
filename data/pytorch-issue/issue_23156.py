import torch
import torch.nn as nn
x = torch.ones(3, 4)
linear = nn.Linear(4, 2)
def has_err(x):
    return bool(((x != x) | (x == float("inf")) | (x == float("-inf"))).any().item())

linear(x).sum().backward()
print(linear.weight.grad)
# tensor([[3., 3., 3., 3.],
#         [3., 3., 3., 3.]])
assert not has_err(linear.weight.grad)

x[2].fill_(float("nan"))
print(linear(x))
# tensor([[ 0.0570, -1.0066],
#         [ 0.0570, -1.0066],
#         [    nan,     nan]], grad_fn=<AddmmBackward>)

linear.zero_grad()
linear(x).sum().backward()
print(linear.weight.grad)
# tensor([[nan, nan, nan, nan],
#         [nan, nan, nan, nan]])
assert has_err(linear.weight.grad)

linear.zero_grad()
linear(x)[:2].sum().backward()
assert not has_err(linear.weight.grad)  # AssertionError raised

x = torch.ones(3, 4)
x[2].fill_(0.5)
linear.zero_grad()
linear(x).sum().backward()
print(linear.weight.grad)
# without masking:
# tensor([[2.5000, 2.5000, 2.5000, 2.5000],
#         [2.5000, 2.5000, 2.5000, 2.5000]])
linear.zero_grad()
linear(x)[2:].sum().backward()
print(linear.weight.grad)
# with masking:
# tensor([[2., 2., 2., 2.],
#         [2., 2., 2., 2.]])

import torch
import torch.nn as nn
x = torch.ones(3, 4, device='cuda')
linear = nn.Linear(4, 2).to(torch.device('cuda'))
def has_err(x):
    return bool(((x != x) | (x == float("inf")) | (x == float("-inf"))).any().item())

linear.zero_grad()
linear(x)[:2].sum().backward()
assert not has_err(linear.weight.grad)  # AssertionError raised