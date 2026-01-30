import torch

def test_1(input_x):
    y = input_x.abs().max()

    z = input_x / 10.0
    z_t = z.t().contiguous().t()  # `z` and `z_t` will be fused into a tiled pointwise

    return y, z, z_t

def test_2(x):
    y = x.abs().max(dim=-1)
    z = x.abs().max()   # we want the first-level reduction of `z` can be fused with `y`.
    return y[0], z

def test_1(input_x):
    y = input_x.abs().max()

    z = input_x / 10.0
    z_t = z.t().contiguous().t()  # `z` and `z_t` will be fused into a tiled pointwise

    return y, z, z_t

test = torch.compile(test)
x = torch.randn(3072, 4096, device="cuda") / 10.0
y, z, z_t = test(x)

def test_2(x):
    y = x.abs().max(dim=-1)
    z = x.abs().max()   # we want the first-level reduction of `z` can be fused with `y`.
    return y[0], z

test = torch.compile(test)
x = torch.randn(3072, 4096, device="cuda")
z = test(x)

def test(x):
        z = x.abs().max()
        return z

test = torch.compile(test)
x = torch.randn(3072, 4096, device="cuda")
z = test(x)

def test_1(input_x):
    y = input_x.abs().max()

    z = input_x / 10.0
    z_t = z.t().contiguous().t()  # `z` and `z_t` will be fused into a tiled pointwise

    return y, z, z_t

test = torch.compile(test)
x = torch.randn(3072, 4096, device="cuda") / 10.0
y, z, z_t = test(x)

(numel1, 1),
(numel2, rnumel2, 1),