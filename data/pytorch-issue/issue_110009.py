import torch
A = torch.rand(5,10)
A.requires_grad=True

k=2

####### torch.linalg.pinv
b = A.t()@A
u,d,v = torch.svd(b)
pinv = torch.linalg.pinv(b,rtol = (d[k]/d[0]).detach())
print(pinv.sum())
loss = pinv.sum()
loss.backward()
print(A.grad)
A.grad.zero_()

####### torch.svd
b = A.t()@A
u,d,v = torch.svd(b)
d_new = torch.zeros_like(d)
d_new[:k] = 1 / d[:k]
pinv_svd = u@torch.diag(d_new)@v.t()
print(pinv_svd.sum())
loss = pinv_svd.sum()
loss.backward()
print(A.grad)

import torch

print(torch.__version__)
torch.manual_seed(1)
A = torch.rand(5, 10)
A.requires_grad = True
b = A.t() @ A
u, d, v = torch.svd(b)
sv_diffs = ((d.view(-1, 1) - d.view(1, -1)) +
            (torch.eye(d.shape[0]) * 99999))
print("Min singular value diff", sv_diffs.abs().min().item())
print("Smallest singular value", d.min().item())
k = 2

# torch.linalg.pinv
b = A.t() @ A
u, d, v = torch.svd(b)
pinv = torch.linalg.pinv(b, rtol=(d[k] / d[0]).detach())
loss = pinv.sum()
loss.backward()
pinv_grad = A.grad.clone().detach()
A.grad.zero_()

# torch.svd
b = A.t() @ A
u, d, v = torch.svd(b)
d_new = torch.zeros_like(d)
d_new[:k] = 1 / d[:k]
pinv_svd = u @ torch.diag(d_new) @ v.t()
loss = pinv_svd.sum()
loss.backward()
svd_grad = A.grad.clone().detach()
A.grad.zero_()

print("Grad max diff", (pinv_grad - svd_grad).abs().max().item())
print("Grad median diff", (pinv_grad - svd_grad).abs().median().item())
print("Max diff", (pinv.sum() - pinv_svd.sum()).abs().max().item())

import torch

A = torch.rand(5,10).double()
B = A.detach().clone()
B[0][0] = B[0][0] + 0.00001
A.requires_grad=True
B.requires_grad=True
k=2

##### (f(x1)-f(x)) / (x1-x) for torch.linalg.pinv
b = A.t()@A
u,d,v = torch.svd(b)
pinv_A = torch.linalg.pinv(b,rtol = (d[k]/d[0]).detach())

b = B.t()@B
u,d,v = torch.svd(b)
pinv_B = torch.linalg.pinv(b,rtol = (d[k]/d[0]).detach())
grad_pinv_manual = (pinv_A.sum()-pinv_B.sum())/(A[0][0] - B[0][0])

##### (f(x1)-f(x)) / (x1-x)for torch.svd
b = A.t()@A
u,d,v = torch.svd(b)
d_new = torch.zeros_like(d)
d_new[:k] = 1 / d[:k]
pinv_svd_A = u@torch.diag(d_new)@v.t()

b = B.t()@B
u,d,v = torch.svd(b)
d_new = torch.zeros_like(d)
d_new[:k] = 1 / d[:k]
pinv_svd_B = u@torch.diag(d_new)@v.t()
grad_svd_manual = (pinv_svd_A.sum()-pinv_svd_B.sum())/(A[0][0] - B[0][0])


####### torch.linalg.pinv
b = A.t()@A
u,d,v = torch.svd(b)
pinv = torch.linalg.pinv(b,rtol = (d[k]/d[0]).detach())
loss = pinv.sum()
loss.backward()
grad_pinv = A.grad[0][0].clone().detach()
A.grad.zero_()

####### torch.svd
b = A.t()@A
u,d,v = torch.svd(b)
d_new = torch.zeros_like(d)
d_new[:k] = 1 / d[:k]
pinv_svd = u@torch.diag(d_new)@v.t()
loss = pinv_svd.sum()
loss.backward()
grad_svd = A.grad[0][0].clone().detach()
print('grad_pinv_manual', grad_pinv_manual)
print('grad_svd_manual', grad_svd_manual)
print('grad_pinv', grad_pinv)
print('grad_svd', grad_svd)
print("Grad diff", (grad_pinv - grad_svd).abs().item())