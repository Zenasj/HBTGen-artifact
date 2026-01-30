import torch

t = torch.rand((1, 2,))
print(t)
v, _ = torch.topk(t, 1)
t[t < v[:, [-1]]] = -float('Inf')
print(t)  # the lower logit is correctly set to -inf

t = torch.rand((1, 2,)).to('mps')
print(t)
v, _ = torch.topk(t, 1)
t[t < v[:, [-1]]] = -float('Inf')
print(t)  # the lower logit has been replaced with the value of `v` instead

3
import torch

t = torch.rand((1, 2,))
t_mps = t.detach().clone().to('mps')
print(t)
v, _ = torch.topk(t, 1)
t[t < v[:, [-1]]] = -float('Inf')
print(t)  # the lower logit is correctly set to -inf

t = t_mps
print(t)
v, _ = torch.topk(t, 1)
t[t < v[:, [-1]]] = -float('Inf')
print(t)  # the lower logit has been replaced with the value of `v` instead