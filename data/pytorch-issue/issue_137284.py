import torch
## Current result
S = torch.tensor([[1.,1.],[1.,2.]], requires_grad=True)
A = torch.tril(torch.ones((2,2)),-1)
# >> A
# tensor([[0., 0.],
#         [1., 0.]])
L = torch.linalg.cholesky(S)
loss = (L*A).sum()
loss.backward(reta)
# >> S.grad[0,1]
# tensor(0.5000)

## What we should get if we manually compute L =cholesky(S)
S = torch.tensor([[1.,1.],[1.,2.]], requires_grad=True)
manual_L = torch.stack([torch.cat([S[0][:1]**.5, torch.tensor([0.0])]),
    torch.cat([S[0][1:]/S[0][:1]**.5, (S[1][1:] - S[0][1:]**2/S[0][:1])**.5])])
true_loss = torch.sum(e_L*A)
true_loss.backward()
# >> S.grad[0,1]
# tensor(1.)

import torch
torch.manual_seed(2024)
C=torch.randn(4,4)/4.0
S = C.t().mm(C)
S.requires_grad_()
A = torch.randn(4,4)/4.0
loss = (A*torch.linalg.cholesky(S)).sum()
loss.backward()
S.grad
## current grad
tensor([[4.8560, 5.0956, 4.2147, 2.6087],
        [5.0956, 4.6948, 3.8186, 2.2791],
        [4.2147, 3.8186, 3.4634, 1.8416],
        [2.6087, 2.2791, 1.8416, 0.7801]])

## proposed pull request
tensor([[ 4.8560, 10.1912,  8.4293,  5.2173],
        [10.1912,  4.6948,  7.6372,  4.5582],
        [ 8.4293,  7.6372,  3.4634,  3.6832],
        [ 5.2173,  4.5582,  3.6832,  0.7801]])