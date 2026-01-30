import torch

from torch import ones, float32
from einops import rearrange
from torch.nn import Linear

to_q = Linear(768, 320, bias=False, device='mps')
to_k = Linear(768, 320, bias=False, device='mps')
to_v = Linear(768, 320, bias=False, device='mps')

context = ones([1, 77, 768], device='mps', dtype=float32)
q = to_q(context)
k = to_k(context)
v = to_v(context)
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=8), (q, k, v))
print(q.isnan().sum())
# tensor(21440, device='mps:0')
print(q.isnan().sum())
# tensor(0, device='mps:0')

print(q.detach().isnan().sum())

print(q.contiguous().isnan().sum())

print((q+0).isnan().sum())