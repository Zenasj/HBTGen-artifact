import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(backend="aot_eager", fullgraph=True)
def f(x):
    y = x.item()
    torch._check_is_size(y)
    if y >= 0:
        return x * 2
    else:
        return x * 3

print(f(torch.tensor([3])))
print(f(torch.tensor([-2])))

tensor([6])
tensor([-4])