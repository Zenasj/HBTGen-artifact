import torch.nn as nn

import torch
import torch.nn.functional as F

t1 = torch.arange(20).float().reshape(5,4)
n1 = torch.nested.as_nested_tensor([t1[:2], t1[2:5]], layout=torch.jagged)

t2 = t1 * 10
n2 = torch.nested.as_nested_tensor([t2[:1], t2[1:5]], layout=torch.jagged)

n1g = n1.clone().detach().requires_grad_()
tensor = F.scaled_dot_product_attention(query=n1g.unsqueeze(2).transpose(1,2), key=n2.unsqueeze(2).transpose(1,2), value=n2.unsqueeze(2).transpose(1,2))

loss  = tensor.values().sum()

### RuntimeError: The function '_nested_view_from_jagged' is not differentiable with respect to argument 'min_seqlen'. This input cannot have requires_grad True.
grad = torch.autograd.grad(loss, n1g, create_graph=True)[0]

### Works
grad = torch.autograd.grad(loss, n1g, create_graph=False)[0]