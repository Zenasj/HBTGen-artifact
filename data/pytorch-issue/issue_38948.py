3
# lob.py
import torch as T
T.autograd.set_detect_anomaly(True)

A = T.randn(10, 10)
A.requires_grad_()
S = A.matmul(A.t())
e, v = T.lobpcg(S, k=3)
S_hat = T.einsum('ij,j,kj->ik', v, e, v) # v * diag(e) * v^T
loss = S_hat.abs().sum()
loss.backward() # breaks here