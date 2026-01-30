import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(fullgraph=True)
def f(a, b):
    xs = b.tolist()
    for x in xs:
        torch._check_is_size(x)
        torch._check(x <= 20)
    return a.split(xs)

N = 2

splits = torch.randint(10, (N,))
sz = splits.sum().item()
            
f(torch.randn(sz), splits)