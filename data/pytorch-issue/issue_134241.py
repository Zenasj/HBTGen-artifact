import torch
import torch._dynamo

torch._dynamo.config.capture_dynamic_output_shape_ops = True


def test(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(y, dtype=z.dtype)
    x[y] = z[y]
    x = x*2
    return x

z = torch.randn(10, 10, requires_grad=True).cuda()
y = (torch.randint(0, 1, (10, 10)) == 1).cuda()

test_comp = torch.compile(test, mode='reduce-overhead', fullgraph=True)
x_comp = test_comp(y, z)
loss_comp = x_comp.sum()
loss_comp.backward()

import torch
import torch._dynamo

torch._dynamo.config.capture_dynamic_output_shape_ops = True


def test(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(y, dtype=z.dtype)
    x[y] = z[y]
    x = x*2
    return x

z = torch.randn(10, 10, requires_grad=True).cuda()
y = (torch.randint(0, 2, (10, 10)) == 1).cuda()

# test = torch.compile(test, mode='reduce-overhead', fullgraph=True)
x_comp = test(y, z)
loss_comp = x_comp.sum()
loss_comp.backward()
print(loss_comp)


def goodtest(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(y, dtype=z.dtype).reshape(-1)
    x[y.reshape(-1).nonzero().squeeze()] = z.reshape(-1)[y.reshape(-1).nonzero().squeeze()]
    x = x*2
    return x.reshape_as(y)

goodtest_comp = torch.compile(goodtest, mode='default', fullgraph=True)
x_comp = goodtest_comp(y, z)
loss_comp = x_comp.sum()
loss_comp.backward()
print(loss_comp)