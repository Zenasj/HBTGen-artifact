import torch

def fn(x):
    z = torch.zeros(3)
    return torch.func.grad(lambda x: (z @ x))(x)

x = torch.zeros(3)

# jvp - succeeds
tangents = torch.ones(3)
output, jvp = torch.func.jvp(fn, (x,), (tangents,))

# linearize - fails
output, jvp_fn = torch.func.linearize(fn, x)