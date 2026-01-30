import torch

def fn(a, b):
    return a.shape[0] * a * b

arg1 = torch.randn(4, 3)
torch._dynamo.mark_dynamic(arg1, 0, min=2, max=10)

compiled_fn = torch.compile(fn)
out = compiled_fn(arg1, torch.randn(4, 3))

new_arg1 = torch.randn(8, 3)
out = compiled_fn(new_arg1, torch.randn(8, 3))