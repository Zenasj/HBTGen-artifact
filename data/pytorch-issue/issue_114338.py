import torch

def fn(x):
    with torch.enable_grad():
        out = x + 1
    return out
    

opt_fn = torch.compile(fn, backend="inductor")  # also, aot_eager, aot_eager_decomp_partition, but not eager
with torch.no_grad():
    res = fn(torch.zeros(10, requires_grad=True))
    opt_res = opt_fn(torch.zeros(10, requires_grad=True))

    assert res.requires_grad == opt_res.requires_grad  # True != False

def fn(x):
    with torch.no_grad():
        out = x + 1
    return out
    

opt_fn = torch.compile(fn, backend="inductor")
with torch.enable_grad():
    res = fn(torch.zeros(10, requires_grad=True))
    opt_res = opt_fn(torch.zeros(10, requires_grad=True))

    assert res.requires_grad == opt_res.requires_grad  # False == False

def fn(x):
    with torch.enable_grad():
        out = x + 1
    out = out + 1
    return out
    

opt_fn = torch.compile(fn, backend="aot_eager")
with torch.no_grad():
    res = fn(torch.zeros(10, requires_grad=True))
    opt_res = opt_fn(torch.zeros(10, requires_grad=True))

    assert res.requires_grad == opt_res.requires_grad  # True != False

BACKEND = "inductor"

def fn(x):
    with torch.enable_grad():
        out = x + 1
    out2 = out + 1
    return out, out2
    

opt_fn = torch.compile(fn, backend=BACKEND)

inp = torch.ones(10, requires_grad=True)
opt_inp = torch.ones(10, requires_grad=True)

with torch.no_grad():
    res = fn(inp)
    opt_res = opt_fn(opt_inp)

    assert all(res.requires_grad == ref for res, ref in zip(res, (True, False))) 
    assert all(res.requires_grad == opt.requires_grad for res, opt in zip(res, opt_res)) 

def fn(x):
    with torch.no_grad():
        out = x + 1
    out2 = x + 1
    return out, out2
    

opt_fn = torch.compile(fn, backend=BACKEND)

inp = torch.ones(10, requires_grad=True)
opt_inp = torch.ones(10, requires_grad=True)

with torch.enable_grad():
    res = fn(inp)
    opt_res = opt_fn(opt_inp)

    assert all(res.requires_grad == ref for res, ref in zip(res, (False, True))) 
    assert all(res.requires_grad == opt.requires_grad for res, opt in zip(res, opt_res))

BACKEND = "inductor"

def fn(x):
    with torch.enable_grad():
        out = x + 1
    out2 = x + 1
    return out, out2

opt_fn = torch.compile(fn, backend=BACKEND)

inp = torch.ones(10, requires_grad=True)
opt_inp = torch.ones(10, requires_grad=True)

with torch.no_grad():
    res = fn(inp)
    opt_res = opt_fn(opt_inp)

    assert all(id(a) != id(b) for a, b in zip(res, res[1:]))
    assert all(id(a) != id(b) for a, b in zip(opt_res, opt_res[1:]))  # fails

    assert all(res.requires_grad == ref for res, ref in zip(res, (True, False))) 
    assert all(res.requires_grad == opt.requires_grad for res, opt in zip(res, opt_res))