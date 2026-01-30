import torch
# Example tensor
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# Scalar tensor indicating the index
index = torch.tensor(1, dtype=torch.int64)

@torch.compile(fullgraph=True, dynamic=True)
def f(x, index):
    idx = index.item()
    torch._check(idx >= 0)
    torch._check(idx < x.size(0))
    return x[idx]

torch._dynamo.config.capture_scalar_outputs = True
f(A, index)

import torch
# Example tensor
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# Scalar tensor indicating the index

@torch.compile(fullgraph=True)
def f(x):
    index = torch.ones([], dtype=torch.int64)
    idx = index.item()
    torch._check(idx >= 0)
    torch._check(idx < x.size(0))
    return x[idx]

torch._dynamo.config.capture_scalar_outputs = True
f(A)