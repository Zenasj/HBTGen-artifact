torch.add(a, b, out=a) # (1) a in-placed
torch.add(a, b, out=b) # (2) b in-placed

import torch

torch.manual_seed(4)
a = torch.randn(4, 4)
b = torch.randn(4, 4)
b.fill_(1.0)

a_mkl = a.to_mkldnn()
b_mkl = b.to_mkldnn()

torch.add(b, a, alpha=1.0, out=a)
torch.add(b_mkl, a_mkl, alpha=1.0, out=a_mkl)

print(a)
print(a_mkl)

tensor([[ 0.0586,  2.2632,  0.8162,  1.1505],
        [ 1.1075,  0.7220, -1.6021,  1.6245],
        [ 0.1316,  0.7949,  1.3976,  1.6699],
        [ 0.9463,  1.0467, -0.7671, -1.1205]])
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]], layout=torch._mkldnn)

tensor([[ 0.0586,  2.2632,  0.8162,  1.1505],
        [ 1.1075,  0.7220, -1.6021,  1.6245],
        [ 0.1316,  0.7949,  1.3976,  1.6699],
        [ 0.9463,  1.0467, -0.7671, -1.1205]])
tensor([[ 0.0586,  2.2632,  0.8162,  1.1505],
        [ 1.1075,  0.7220, -1.6021,  1.6245],
        [ 0.1316,  0.7949,  1.3976,  1.6699],
        [ 0.9463,  1.0467, -0.7671, -1.1205]], layout=torch._mkldnn)