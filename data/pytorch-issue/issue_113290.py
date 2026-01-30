import torch

@torch.compile(backend="aot_eager")
def fn(x, s):
    x.requires_grad = True
    y = x * 2
    x.requires_grad = False
    z = x * 3 * s * y
    return x, y, z

x = torch.ones(3)
s = torch.ones(3, requires_grad=True)

_, y, z = fn(x, s)

x.requires_grad = True
z.sum().backward()

print(x.grad)  # Should be torch.tensor([6., 6., 6.]) via y, but got None