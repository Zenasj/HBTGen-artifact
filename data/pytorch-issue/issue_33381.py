import torch

def f(A, b):
    return torch.lu_solve(b, *torch.lu(A))

A = torch.eye(3)
b = torch.ones((3, 1))

torch.jit.trace(f, (A, b))  # <-- OK
torch.jit.script(f)         # <-- Unknown builtin op: aten::lu

lu_trace = torch.jit.trace(torch.lu, torch.eye(2))

@torch.jit.script
def f(A, b):
    return torch.lu_solve(b, *lu_trace(A))  # <-- OK