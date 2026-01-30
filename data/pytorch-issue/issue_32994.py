import torch
import torch.nn.functional as F

t = torch.randint(1,10,(10,3))
t_nonzero = torch.nonzero(F.pad(t,(0,10)))
as_tuple  = t[(t_nonzero[:,0], t_nonzero[:,1])]

t = torch.randint(1,10,(10,3))
as_tuple = t[torch.nonzero(F.pad(t,(0,10)), as_tuple=True)]

torch.nonzero(t)

t.nonzero()