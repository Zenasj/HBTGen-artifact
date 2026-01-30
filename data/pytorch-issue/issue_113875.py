import torch
def fn(x):
    if x.size() != (5, 1, 2, 3):
        return x.cos()
    return x.sin()

opt_fn = torch.compile(fn, backend="eager", dynamic=True)

x = torch.ones(5, 1, 3, 4)
x2 = torch.ones(5, 1, 2, 3)
torch.testing.assert_close(fn(x), opt_fn(x))
# Installs guard: ~(Eq(L['x'].size()[0], 5) & Eq(L['x'].size()[2], 2) & Eq(L['x'].size()[3], 3))
torch.testing.assert_close(fn(x2), opt_fn(x2))
# ~(Eq(L['x'].size()[0], 5) & Eq(L['x'].size()[2], 2) & Eq(L['x'].size()[3], 3)) should fail, but doesn't!