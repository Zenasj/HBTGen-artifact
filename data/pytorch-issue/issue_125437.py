import torch

@torch.compile(mode="max-autotune")
def foo(x, y):
    return x @ y


x = torch.empty_strided((50257, 32768), ((1, 50304)), dtype=torch.bfloat16, device='cuda')
y = torch.empty_strided((32768, 768), (768, 1), dtype=torch.bfloat16, device='cuda')

foo(x, y)

x = torch.empty_strided((50257, 32768), ((1, 50304)), dtype=torch.bfloat16, device='cuda')

ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)

ram = rm % M

ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)