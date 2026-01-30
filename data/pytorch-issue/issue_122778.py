import torch

m = torch.load(f'./gate_linear.pt',map_location='cpu')
print(m)
#Linear(in_features=1024, out_features=2816, bias=False)

x1 = torch.load(f'./x1.pt',map_location='cpu')
x2 = torch.load(f'./x2.pt',map_location='cpu')
print(x1.size(),x2.size())
#torch.Size([1, 128, 1024]) torch.Size([1, 129, 1024])

print(torch.equal(x1,x2[:,:-1,:]))
#True

o1 = m(x1)
o2 = m(x2)

print(torch.equal(
    o1,
    o2[:,:-1,:]
    ))

#False

import torch
torch.set_printoptions(precision=8)
x = torch.tensor([[[ 0.0451, -0.8093],
                   [-0.3275, -0.5304]]]
                 )
x1 = x[:,:-1,:]
x2 = x
w = torch.tensor(   [[ 0.29,  0.05],
                     [-0.11, -0.29],
                     [ 0.05,  0.29],
                     [-0.03, -0.05]], requires_grad=True)
w = w.transpose(0,1)

print(x.size(),x.dtype)
print(w.size(),w.dtype)
#torch.Size([1, 2, 2]) torch.float32
#torch.Size([2, 4]) torch.float32

o1 = torch.matmul(x1,w)
o2 = torch.matmul(x2,w)

print(o1)
print(o2)
# tensor([[[-0.02738600,  0.22973600, -0.23244201,  0.03911200],
#          [-0.12149499,  0.18984099, -0.17019099,  0.03634500]]],
#        grad_fn=<UnsafeViewBackward0>)
print(torch.equal(o1,o2[:,:-1,:]))
#False