import torch
def fn(x):
    result = torch.rrelu(x,0.2,0.8,training=True)
    return result
x = torch.randn(4,4,dtype=torch.bfloat16,requires_grad=True)
compiled_fn = torch.compile(fn, backend="inductor")

res = compiled_fn(x)