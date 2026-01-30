import torch

def fn(x):
    t = x.pow(5) # 4 ^ 5 = 1024 -> 0 (uint8)
    return torch.cos(t) # should be 1
    # More examples
    # ❌ torch.sin, torch.erf, torch.sigmoid
    # ✔️ torch.relu, torch.add(_, 1)

x = torch.tensor([4], dtype=torch.uint8) # ✔️ x is cuda tensor.
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
tensor([1.])
>>> ret_compiled
tensor([0.9874])
"""