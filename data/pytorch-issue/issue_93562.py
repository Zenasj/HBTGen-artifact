import torch


def foo(x):
    return torch.ops.aten.select.int(x, 1, 2)

foo_opt = torch._dynamo.optimize("inductor")(foo)

y = torch.rand([3, 4, 5, 6])
z = y[2]

out_opt = foo_opt(z)
out_eager = foo(z)
print(out_opt.storage_offset() == out_eager.storage_offset())  
print(torch.allclose(out_opt, out_eager))

True
True