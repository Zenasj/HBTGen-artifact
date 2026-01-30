import torch.nn as nn

python
import copy
import timeit
import torch
from torch import nn

# Before self-attention (some computation)
c = torch.randn(1024, 2, 256, device='cuda')
d = torch.randn(1024, 2, 256, device='cuda')

t1 = timeit.timeit('torch.eq(c, d).all()', number=1, globals=globals())*1e3
t2 = timeit.timeit('torch.eq(c, d).all()', number=1, globals=globals())*1e3
t3 = timeit.timeit('torch.eq(c, d).all()', number=1, globals=globals())*1e3

print(f"First time with eq/all (before self-attention): {t1: .4f} ms")
print(f"Second time with eq/all (before self-attention): {t2: .4f} ms")
print(f"Third time with eq/all (before self-attention): {t3: .4f} ms\n")

t1 = timeit.timeit('torch.equal(c, d)', number=1, globals=globals())*1e3
t2 = timeit.timeit('torch.equal(c, d)', number=1, globals=globals())*1e3
t3 = timeit.timeit('torch.equal(c, d)', number=1, globals=globals())*1e3

print(f"First time with equal (before self-attention): {t1: .4f} ms")
print(f"Second time with equal (before self-attention): {t2: .4f} ms")
print(f"Third time with equal (before self-attention): {t3: .4f} ms\n")

# Perform self-attention (some computation)
a = torch.randn(1024, 2, 256, device='cuda')
b = torch.randn(1024, 2, 256, device='cuda')

self_attention = nn.MultiheadAttention(256, 8).to('cuda')
layers = nn.ModuleList([copy.deepcopy(self_attention) for _ in range(20)])

for layer in layers:
    a = layer(a+b, a+b, a)[0]

# After self-attention (some computation)
t1 = timeit.timeit('torch.eq(c, d).all()', number=1, globals=globals())*1e3
t2 = timeit.timeit('torch.eq(c, d).all()', number=1, globals=globals())*1e3
t3 = timeit.timeit('torch.eq(c, d).all()', number=1, globals=globals())*1e3

print(f"First time with eq/all (after self-attention): {t1: .4f} ms")
print(f"Second time with eq/all (after self-attention): {t2: .4f} ms")
print(f"Third time with eq/all (after self-attention): {t3: .4f} ms\n")

t1 = timeit.timeit('torch.equal(c, d)', number=1, globals=globals())*1e3
t2 = timeit.timeit('torch.equal(c, d)', number=1, globals=globals())*1e3
t3 = timeit.timeit('torch.equal(c, d)', number=1, globals=globals())*1e3

print(f"First time with equal (after self-attention): {t1: .4f} ms")
print(f"Second time with equal (after self-attention): {t2: .4f} ms")
print(f"Third time with equal (after self-attention): {t3: .4f} ms\n")

python
import copy
import timeit
import torch
from torch import nn

a = torch.randn(1024, 2, 256, device='cuda')
b = torch.randn(1024, 2, 256, device='cuda')
_ = torch.matmul(a.transpose(1, 2), b)

self_attention = nn.MultiheadAttention(256, 8).to('cuda')
layers = nn.ModuleList([copy.deepcopy(self_attention) for _ in range(20)])


def run_test(a, b, layers):
    for layer in layers:
        a = layer(a+b, a+b, a)[0]


t = timeit.timeit('run_test(a, b, layers)', number=1, globals=globals())*1e3
print(f"Recorded timeit time: {t: .4f} ms")