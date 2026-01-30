import torch
torch.set_default_device('cuda')

def f(x):
    z = x + 1
    z = torch.view_as_complex(z)
    a = torch.view_as_real(z)
    out = a + 1
    return out, torch.view_as_real(z + 1)

print(f(torch.zeros(4, 2)))
print(torch.compile(f)(torch.zeros(4, 2)))