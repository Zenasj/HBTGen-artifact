import torch

a = torch.ones(2)
b = a.view(2)
b.resize_(4) # try resize a **view** tensor to a larger size

a = torch.ones(2)
a_view = a.view(2)
a.resize_(4)