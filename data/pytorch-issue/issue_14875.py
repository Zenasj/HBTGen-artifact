py
import torch

x = torch.tensor(2.)

def f1(x):
    return x, x

jit_f1 = torch.jit.trace(f1, x)

print(f1(x))  # (tensor(2.), tensor(2.))
print(jit_f1(x))  # (tensor(2.), tensor(2.))
assert f1(x) == jit_f1(x)  # ok


def f2(x):
    return (x,)

jit_f2 = torch.jit.trace(f2, x)

print(f2(x))  # (tensor(2.),)
print(jit_f2(x))  # tensor(2.)
assert f2(x) == jit_f2(x)  # fails