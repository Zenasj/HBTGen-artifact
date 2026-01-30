import torch

@torch.compile(mode='max-autotune', dynamic=False)
def func(inp, mat1, mat2):
    res = torch.addmm(inp, mat1, mat2)
    return res

inp = torch.randn(128, device='cuda')
mat1 = torch.randn(16, 64, device='cuda')
mat2 = torch.randn(64, 128, device='cuda')
res = func(inp, mat1, mat2)
print('res', res)

# change size of mat1
mat1 = torch.randn(32, 64, device='cuda')
res = func(inp, mat1, mat2)
print('res', res)