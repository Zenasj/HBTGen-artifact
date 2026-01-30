import torch.nn as nn

py
import torch

torch.manual_seed(420)

class M(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(M, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin1(x)
        return x

in_dim = 0
out_dim = 1

func = M(in_dim, out_dim).to('cpu')

in_dim = 0
x = torch.randn(32, in_dim)

with torch.no_grad():
    func = func.eval()
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # Intel MKL ERROR: Parameter 9 was incorrect on entry to cblas_sgemm_pack.
    # [1] floating point exception (core dumped)