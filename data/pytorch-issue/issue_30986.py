import torch

def run(dtype):
    a = torch.ones((5,), device='cpu', dtype=dtype)
    b = torch.empty((5, 0), device='cpu', dtype=dtype)
    c = torch.empty((0,), device='cpu', dtype=dtype)
    print(a.addmv(b, c, alpha=1, beta=3))

run(torch.float)
run(torch.int)