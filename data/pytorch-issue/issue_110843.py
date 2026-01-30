import torch

@torch.compile(backend="aot_eager", fullgraph=True)
def f(flat_p, p):
    with torch.no_grad():
        p.set_(flat_p[0:2])
    return flat_p

flat_p = torch.randn(4, requires_grad=True)
f(flat_p, flat_p[0:2])

def composite_set_(x, y):
    if storages_alias(x, y) and same_metadata(x, y):
        # no-op
        return
    # we'll need to make sure there's a reasonable way to run the initial set_() implementation
    # (maybe by disabling the python dispatcher?)
    with disable_python_dispatcher():
        return x.set_(y)