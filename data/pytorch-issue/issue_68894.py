import torch as ch

table = ch.zeros(100, 100)
x = ch.randn(100, 100)
inds = ch.arange(100)

print(a_table[inds])
table[inds].copy_(x, non_blocking=False)
print(a_table[inds])