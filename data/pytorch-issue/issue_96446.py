import torch
a = torch.rand(1) 
b = torch.rand(2)
def forward(a, b):
    a[0] = 2
    return a * b
fn_compiled = torch.compile(forward)
print(fn_compiled(a, b))