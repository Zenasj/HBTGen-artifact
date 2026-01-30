import torch
import torch.nn as nn

def fun(module,grad_in,grad_out):
    print('grad_in')
    print([_grad_in.shape for _grad_in in grad_in if _grad_in is not None])     # add break point here
    print('grad_out')
    print([_grad_out.shape for _grad_out in grad_out if _grad_out is not None])     # add break point here

net = nn.Sequential(nn.Conv2d(1,1,3,), nn.ReLU(), nn.Conv2d(1,1,3,))

net[0].register_backward_hook(fun)
net[2].register_backward_hook(fun)

x = torch.randn(1,1,15,15,requires_grad=True)
l = net(x)

l.backward(torch.ones_like(l))

net = nn.Sequential(nn.Conv1d(1,1,3,), nn.ReLU(), nn.Conv1d(1,1,3,))

net[0].register_backward_hook(fun)
net[2].register_backward_hook(fun)

x = torch.randn(1,1,15,requires_grad=True)
l = net(x)

l.backward(torch.ones_like(l))

grad_in
[torch.Size([1, 1, 13, 13]), torch.Size([1, 1, 3, 3]), torch.Size([1])]
grad_out
[torch.Size([1, 1, 11, 11])]
grad_in
[torch.Size([1, 1, 15, 15]), torch.Size([1, 1, 3, 3]), torch.Size([1])]
grad_out
[torch.Size([1, 1, 13, 13])]

grad_in
[torch.Size([1, 1, 1, 11])]
grad_out
[torch.Size([1, 1, 11])]
grad_in
[torch.Size([1, 1, 1, 13])]
grad_out
[torch.Size([1, 1, 13])]