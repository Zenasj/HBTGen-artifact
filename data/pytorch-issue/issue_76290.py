import torch
import torch.nn as nn
from torch.autograd import grad

batch_sz = 10
inp_sz = 5
z_sz = 2

f = nn.Sequential(nn.Linear(inp_sz, 50),
                  nn.Sigmoid(),
                  nn.Linear(50,z_sz))

x = torch.randn(batch_sz,inp_sz).requires_grad_()
y = f(x)

v = torch.zeros(batch_sz,z_sz, dtype=torch.float)
v[:,0] = 1.
v = v.requires_grad_()

dy = grad(y,x,grad_outputs=v, create_graph=True, is_grads_batched=True)[0]