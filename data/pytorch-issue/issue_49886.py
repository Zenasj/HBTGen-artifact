import torch.nn as nn

import torch
import torch.nn.functional as F

random_init = False

x = torch.zeros(15,3)
if random_init:
    x.normal_()
else:
    for i in range(x.size(1)):
        x[i,i] = i+1

# Generate rank-3 matrix
y = torch.zeros(15, 15)
y[:,:5] = x[:,0][:,None]
y[:,5:10] = x[:,1][:,None]
y[:,10:] = x[:,2][:,None]

y.requires_grad_(True)

U, S, V = torch.svd(y)
print(S)

z =  U @ torch.diag_embed(S) @ V

loss = z.sum()
loss.backward()

torch.all(y.grad == y.grad)