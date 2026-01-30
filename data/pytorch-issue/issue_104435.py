import torch

@torch.compile(mode='reduce-overhead')
def foo(x):
    return x * x * x

inp = torch.rand([2], device="cuda")
out = foo(inp).detach()
sum_val_1 = out+out
out2 = foo(inp).detach()
sum_val_2 = out+out
print(sum_val_1, sum_val_2, out2 + out2)
assert  sum_val_1.sum()==sum_val_2.sum()