import torch
seq1 = torch.rand(367,1024)
seq2 = torch.rand(1245,1024)
seq3 = torch.rand(156,1024)
nest_a = torch.nested.nested_tensor([a1, a2, a3], dtype=torch.float, device=device)
nest_e = torch.nested.nested_tensor([torch.rand(1,1024), torch.rand(1,1024), torch.rand(1,1024)], dtype=torch.float, device=device)
nest_a * nest_e

import torch

device = "cuda"

a1 = torch.rand(367,1024)
a2 = torch.rand(1245,1024)
a3 = torch.rand(156,1024)
# shape (3, j1, 1024)
nest_a = torch.nested.nested_tensor(
    [a1, a2, a3],
    dtype=torch.float,
    device=device,
    layout=torch.jagged
)
# shape (3, 1, 1024)
e = torch.randn(3, 1, 1024, device=device)
# broadcasts
nest_a * e