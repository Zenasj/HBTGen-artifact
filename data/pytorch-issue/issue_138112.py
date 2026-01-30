import torch

def nonlocal_test():
    z = 1
    k = 2

    def create_fn():
        def fn(x):
            nonlocal k, z
            k = z
        return fn

    def run_fn(fn, x):
        nonlocal z
        z = 3
        fn(x)
        return x.cos()

    @torch.compile(backend="eager", fullgraph=True)
    def foo(x):
        fn = create_fn()
        return run_fn(fn, x)

    x = torch.randn(2, 3)
    foo(x)
    print(f'{z=} - {k=}')
    assert z == 3
    assert k == 3  # `k` still references the previous value of `z` (1)

nonlocal_test()