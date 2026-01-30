import torch

x = torch.rand(1,4,4).double()
y = torch.rand(2,4,5).float()
z = x @ y

# RuntimeError: expected scalar type Double but found Float

x = torch.rand(1,4,4).double()
y = torch.rand(2,4,1000).float()
z = x @ y

# Works just fine, but z is wrong -- `x.float() @ y` gives a different result