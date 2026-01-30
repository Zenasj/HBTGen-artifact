import torch
import torch._dynamo
from torch.testing import make_tensor
from torch import tensor

def fn(x, y, t):
    return x.index_reduce(dim=0, index=y, source=t, reduce="prod", include_self=True)

x = torch.empty(3, 5).fill_(2).cuda().T
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float, device='cuda')
index = torch.tensor([0, 4, 2, 0]).cuda()
args = [x, index, t]
if True:
    x.requires_grad = True
    t.requires_grad = True
    args = [x, index, t]
print(args)

fn_opt = torch._dynamo.optimize("inductor", nopython=True, dynamic=True)(fn)
fn_opt(*args)