import torch

def _kabsch(P, Q):
    u, _, vt = torch.linalg.svd(torch.matmul(P.T, Q).to(torch.float32))
    s = torch.eye(P.shape[1], device=P.device)
    s[-1, -1] = torch.sign(torch.linalg.det(torch.matmul(u, vt))) # segfault happens here
    r_opt = torch.matmul(torch.matmul(u, s), vt)
    return r_opt.to(device=P.device, dtype=P.dtype)

device = torch.device('cuda:0')
_kabsch(
    torch.randn(512, 3).to(device),
    torch.randn(512, 3).to(device),
)

import torch
device = torch.device('cuda:0')
x = torch.randn(3, 3)
torch.linalg.det(x.to(device)) # segfaults