import torch
a = torch.tensor([1.], device='cuda:0')
b = torch.tensor(0., device='cuda:0')

def forward(a, b):
    backup = a[0]
    if b < 1e5:
        a[0] = backup
        a.max()

with torch.no_grad():
	fn_compiled = torch.compile(forward)
	print(fn_compiled(a, b))