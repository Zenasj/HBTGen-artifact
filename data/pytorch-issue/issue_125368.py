import torch
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True


@torch.compile(fullgraph=True)
def f(x):
    return torch.tensor([x], dtype=torch.int64)


print(f(torch.tensor(20)))