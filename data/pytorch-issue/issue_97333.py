import torch

def fn(x):
    return torch.fmod(x, 2.3)
    # "✔️" means that the program works fine with the substitution.
    # ✔️ return torch.fmod(x, 2.5)

x = torch.tensor(
  [[4.9805, 6.6445, 5.6836, 6.1055],
   [3.4121, 4.1406, 5.1523, 5.9766],
   [3.3691, 4.4102, 6.8008, 4.0156],
   [6.8633, 6.8750, 6.9805, 5.7070]], dtype=torch.float16)
# ✔️ dtype=torch.float32
# ✔️ x = torch.rand([4, 4], dtype=torch.float16)

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
raceback (most recent call last):
  File "repro.py", line 18, in <module>
    assert torch.allclose(
AssertionError: 
>>> ret_eager
tensor([[0.3789, 2.0430, 1.0820, 1.5039],
        [1.1113, 1.8398, 0.5508, 1.3750],
        [1.0684, 2.1094, 2.1992, 1.7148],
        [2.2617, 2.2734, 0.0781, 1.1055]], dtype=torch.float16)
>>> ret_compiled
tensor([[0.3804, 2.0449, 1.0840, 1.5059],
        [1.1123, 1.8408, 0.5522, 1.3770],
        [1.0693, 2.1094, 2.2012, 1.7158],
        [2.2637, 2.2754, 0.0804, 1.1074]], dtype=torch.float16)
"""

import torch

def fn(x):
    return torch.fmod(x, 2.3)

x = torch.tensor([[6.9805]], dtype=torch.float16).cuda()

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