py
import torch

def fn(x):
    output = torch.log(torch.exp((- (x ** 2))))
    output = torch.atleast_1d(output)
    print(output)
    output = output.to(x.dtype)
    return output

x = torch.tensor([1, 2, 3, 4, 5], device=torch.device('cpu'))
print('cpu')
print(fn(x))

print('cuda')
print(fn(x.cuda()))

cpu
tensor([ -1.,  -4.,  -9., -16., -25.])
tensor([ -1,  -4,  -9, -16, -25])
cuda
tensor([ -1.0000,  -4.0000,  -9.0000, -16.0000, -25.0000], device='cuda:0')
tensor([  0,  -4,  -9, -16, -25], device='cuda:0')