import torch
a = torch.rand((1, 2))
b = torch.rand((1, 1))
def forward():
    c = torch.matmul(b, b) # [1,1]
    return torch.add(c, a) # [1,1] + [1,2]
print(forward())
fn_compiled = torch.compile(forward)
print(fn_compiled())