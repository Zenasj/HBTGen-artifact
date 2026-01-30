import torch
def f(x):
    return x*2
z = torch.jit.trace(f, (torch.zeros(10),))
print(type(z))
z.save("out.pt")

@torch.jit.script
def foo(x:int, tup:Tuple[int, int])->int:
    t0, t1 = tup
    return t0 + t1 + x

print(foo(3,(3,13)))
print(type(foo))
torch.jit.save(foo, "test.pt")