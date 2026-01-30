import torch
def forward():
    x = torch.tensor(5, device='cuda', dtype=torch.uint8)
    y = torch.neg(x)
    return x < y
print(forward())
fn_compiled = torch.compile(forward)
print(fn_compiled())

tensor(True, device='cuda:0')
tensor(False, device='cuda:0')