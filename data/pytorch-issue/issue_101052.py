import torch
import torch._dynamo

def fn(x):
    with torch.cuda.amp.autocast(False):
        x = torch.sin(x + 1)
    return x

x = torch.randn([2, 3])
ref = fn(x)
print(ref)
opt_fn = torch._dynamo.optimize(backend="inductor")(fn)
print(opt_fn(x))