import torch

def fn(x):
    with torch.no_grad():
        out = x + 1
    out2 = x + 1  # if not for differing meta, this can (and would) benefit from CSE
    # out2 = x + 2  # no error with this
    return out, out2

opt_fn = torch.compile(fn, backend="aot_eager")

inp = torch.ones(10, requires_grad=True)
opt_inp = torch.ones(10, requires_grad=True)

with torch.enable_grad():
    res = fn(inp)
    opt_res = opt_fn(opt_inp)

    assert all(id(a) != id(b) for a, b in zip(res, res[1:]))
    assert all(id(a) != id(b) for a, b in zip(opt_res, opt_res[1:]))  # fails

    assert all(res.requires_grad == ref for res, ref in zip(res, (False, True))) 
    assert all(res.requires_grad == opt.requires_grad for res, opt in zip(res, opt_res))   # also fails