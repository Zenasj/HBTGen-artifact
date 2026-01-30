import torch

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(fullgraph=True)
def f(x):
    y, y2 = x.tolist()
    torch._check(0 < 1 * y)
    if 2 * y > 0:
        return torch.tensor(True)
    else:
        return torch.tensor(False)

f(torch.tensor([23, 24]))