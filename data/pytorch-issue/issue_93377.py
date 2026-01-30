import torch

torch.manual_seed(0)

device = torch.device('cuda')
x = torch.rand(3).to(device)

def fn1(v4):
    v2 = v4.expand(1, 1, 3)
    return v2

def fn2(v4):
    v2 = v4.expand(1, 1, 3)
    v1 = v4.div_(1.5) # need this line to trigger inconsistency
    return v2

print(fn1(x.cpu().clone()))
print(fn1(x.clone()))
compiled = torch.compile(fn1)
print(compiled(x.clone())) # the same as print(fn1(x.clone()))

print(f'==== finish fn1')

print(fn2(x.clone())) # ERROR! different with other results!
compiled = torch.compile(fn2)
print(compiled(x.clone()))

print(f'==== finish fn2')