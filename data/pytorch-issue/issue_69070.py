import torch
import numpy as np
x = torch.tensor(np.array([[1,2,3]]), requires_grad=True, dtype=torch.float)
y = torch.randn(3) 
y[0] = x[0][0]**2+2*x[0][1]+x[0][2]
y[1] = x[0][0]+x[0][1]**3+x[0][2]**2
y[2] = 2*x[0][0]+x[0][1]**2+x[0][2]**3
torch.autograd.grad(y, x, torch.ones_like(y))

import torch
import numpy as np
x = torch.tensor(np.array([[1,2,3]]), requires_grad=True, dtype=torch.float)
y = torch.randn(3) 
y[0] = x[0][0]**2+2*x[0][1]+x[0][2]
y[1] = x[0][0]+x[0][1]**3+x[0][2]**2
y[2] = 2*x[0][0]+x[0][1]**2+x[0][2]**3
torch.autograd.grad(y, x, (torch.eye(y.shape[0]),), is_grads_batched=True)