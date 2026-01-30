import torch

a = torch.rand([2])
b = torch.rand([2])
def forward(a, b):
    c = torch.cat((a, b))
    return c.min(0).values
print(forward(a, b))
fn_compiled = torch.compile(forward)
print(fn_compiled(a, b))