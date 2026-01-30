import torch
a = torch.randint(4, (2, ), dtype=torch.int16)

def forward(a):
    a = a + 1
    return a.min()

fn_compiled = torch.compile(forward)
print(fn_compiled(a))