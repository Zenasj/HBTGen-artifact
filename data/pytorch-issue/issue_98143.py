import torch

def fn(x):
    # x: (1, 5)
    t1 = torch.add(x, x)
    t2 = t1.unfold(1, 3, 2) # t2: (1, 2, 3)
    t3 = t2.abs_()
    return t3

x = torch.rand([1, 5])
ret_eager = fn(x)
compiled = torch.compile(fn)
ret_compiled = compiled(x)

assert torch.allclose(
    ret_eager, ret_compiled,
    rtol=1e-2, atol=1e-3, equal_nan=True,
), '\n'.join([
    '',
    f'>>> ret_eager',
    str(ret_eager),
    f'>>> ret_compiled',
    str(ret_compiled),
])
print('==== Check OK! ====')

"""
Traceback (most recent call last):
  File "repro.py", line 15, in <module>
    assert torch.allclose(
AssertionError: 
>>> ret_eager
tensor([[[1.8399, 1.4801, 0.2908],
         [0.2908, 0.6970, 1.8828]]])
>>> ret_compiled
tensor([[[1.8399, 1.4801, 0.5817],
         [0.5817, 0.6970, 1.8828]]])
"""