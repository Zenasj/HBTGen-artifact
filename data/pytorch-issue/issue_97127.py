import torch

def forward():
    a = torch.zeros([2, 2])
    b = a.argmax(0)
    if b.float().mean():
        pass

fn_compiled = torch.compile(forward)
print(fn_compiled())