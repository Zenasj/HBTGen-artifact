import torch
def hf0(m, n):
    x0 = torch.ones(m,n,dtype=torch.complex128)
    x1 = torch.ones(m,n,dtype=torch.float64)
    x1[0,0] = torch.inf
    print(x0/x1)
hf0(1, 2)
# tensor([[0.+0.j, 1.+0.j]], dtype=torch.complex128)
hf0(2, 2)
# tensor([[nan+nanj, 1.+0.j],
#         [1.+0.j, 1.+0.j]], dtype=torch.complex128)