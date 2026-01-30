import torch
a = torch.rand(2, requires_grad=True)
b = a + 2
torch.acosh_(b)
b.sum().backward()

import torch
a = torch.rand(2, requires_grad=True)
b = a + 2
torch.acosh_(b)
b.sum().backward()