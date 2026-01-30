import torch
N=150000
X = torch.randn((N,4))
U,S,V = torch.svd(torch.t(X))

import torch
N = 150000
X = torch.randn(N, 4)
X.svd()
X = torch.randn(2 * N, 4)
X.svd()