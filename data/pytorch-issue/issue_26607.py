import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.ones(1, requires_grad=True)
with torch.no_grad():
    y = x.t()
    y.add_(0)

x = torch.ones(3, 2, 1, requires_grad=True)
model = nn.Sequential(
    nn.InstanceNorm1d(2),
    nn.ReLU(inplace=True),
)
y = checkpoint(model, x)
y.norm().backward()

# x.grad should be tensor(...). But it is still None because it didn't receive any gradients.
assert x.grad is not None  # FAILS!

# y.grad_fn should be <CheckpointFunctionBackward>. But it is <AsStridedBackward> actually.
assert y.grad_fn.__class__ is CheckpointFunction._backward_cls  # FAILS!

def test_inplace_view_in_autograd_function(self):
    # See https://github.com/pytorch/pytorch/issues/26546
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = x.t()
            y.add_(0)
            return y

        @staticmethod
        def backward(ctx, g):
            return g.fill_(42)

    x = torch.rand(3, requires_grad=True)
    y = F.apply(x)
    y.sum().backward()

    # y.grad_fn should be <FBackward>, but it is <AsStridedBackward> actually.
    assert y.grad_fn.__class__ is F._backward_cls

    # x.grad.mean() should be 42, but it is 1 actually.
    assert x.grad.mean().item() == 42

x = torch.ones(10, requires_grad=True)
with torch.no_grad():
    y = x[1:]
y.add_(0)

with torch.no_grad():
    b = a.view(a.size())

b = a.detach()