import torch
def forward():
    a = torch.tensor([1])
    a = a[0:1]
    b = a.squeeze()
    a[0] = 0 #(*)
    if a[0] < 1e5:
        pass
    a[0] = 2
    return b

with torch.no_grad():
	print(forward()) # return 2
	fn_compiled = torch.compile(forward)
	print(fn_compiled()) # return 0

import torch
def forward():
    a = torch.rand((2, 4), device='cuda')
    b = a.reshape(2, 2, 2)
    if b.max() >= -1e5:
        a[0, 0] = 0
    return b

with torch.no_grad():
	print(forward().shape) # (2, 2, 2)
	fn_compiled = torch.compile(forward)
	print(fn_compiled().shape) # (2, 4)

tensor(2)
tensor(2)

tensor(2)
tensor([2])

tensor(2)
tensor(0)