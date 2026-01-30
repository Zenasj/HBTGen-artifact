import torch

a = torch.tensor([
    [ 8.3899e-01,  5.4414e-01,  3.3142e+05],
    [-5.4414e-01,  8.3899e-01,  4.6909e+06],
    [ 0.0000e+00,  0.0000e+00,  1.0000e+00]]).type(torch.float)
b = torch.tensor([
    [ 8.3899e-01, -5.4414e-01,  2.2745e+06],
    [ 5.4414e-01,  8.3899e-01, -4.1160e+06],
    [ 0.0000e+00,  0.0000e+00,  1.0000e+00]]).type(torch.float)

print(a @ b)
print(a.cuda() @ b.cuda())
print(a.double().cuda() @ b.double().cuda())
print(a.cuda().double() @ b.cuda().double())