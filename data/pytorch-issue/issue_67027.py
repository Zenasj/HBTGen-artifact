import torch

def wrap(fn):

    def new_fn(*a, **kw):
        return fn(*a, **kw)

    new_fn.inner_fn = fn

    return new_fn


x = torch.ones((2,))

# before wrap
x[0]  # works
x[0:1]  # works
x[torch.tensor([0, 1])]  # works

# wrap
torch.Tensor.__getitem__ = wrap(torch.Tensor.__getitem__)

# after wrap
x[0]  # works
x[0:1]  # works
x[torch.tensor([0, 1])]  # IndexError: too many indices for tensor of dimension 1

# unwrap
torch.Tensor.__getitem__ = torch.Tensor.__getitem__.inner_fn

# after unwrap
x[0]  # works
x[0:1]  # works
x[torch.tensor([0, 1])]  # IndexError: too many indices for tensor of dimension 1