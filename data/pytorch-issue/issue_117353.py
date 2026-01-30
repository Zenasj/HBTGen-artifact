import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)
def f(x, i):
    i0, i1, i2 = i.tolist()
    return torch.functional.split(x, [i0, i1, i2])

print(f(torch.randn(10, requires_grad=True), torch.tensor([3, 3, 4])))