import torch
import torch._dynamo

@torch._dynamo.skip
def f(x, y):
    return x + y

def forward(x, y):
    return f(x, y)

fn_compiled = torch.compile(forward)
x = torch.randn(3)
y = torch.randn(3)
print(fn_compiled(x, y))