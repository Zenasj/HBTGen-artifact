import torch

@torch.compile(backend="inductor")
def fn(x, y):
    return x + y

x = torch.randn(10)
y = torch.randn(10)
print(f"cuda is compiled: {torch.cuda._is_compiled()}")
fn(x, y)