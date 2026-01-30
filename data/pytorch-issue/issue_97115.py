import torch
def forward():
    x = torch.zeros(torch.Size([1]), device='cuda:0')
    def subfunc():
        x[0] = backup

    backup = 1

    if x[0] >= -1e5:
        pass

    subfunc()
    return x

with torch.no_grad():
	print(forward())
	fn_compiled = torch.compile(forward)
	print(fn_compiled())