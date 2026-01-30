import torch

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(fullgraph=True)
def f(x):
    u0, u1 = x.tolist()
    torch._check_is_size(u0)
    if ((u0 + 10) * (u0 + 10)) % (u0 + 10) != 0:
        return torch.tensor(True)
    else:
        return torch.tensor(False)

f(torch.tensor([20, 21]))