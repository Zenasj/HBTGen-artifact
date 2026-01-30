import torch

def fn(x, y):
    _ = y.copy_(x)
    return torch.moveaxis(y, source=0, destination=1)

x = torch.rand([2, 3], dtype=torch.float16)
y = torch.rand([2, 3], dtype=torch.float32) # works fine if x&y has the same type

ret_eager = fn(x, y)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x, y)
print('==== torchcomp mode OK! ====')

"""
==== Eager mode OK! ====
==== torchcomp compilation OK! ====
Traceback (most recent call last):
  File "repro.py", line 15, in <module>
    ret_compiled = compiled(x, y)
  File "python3.10/site-packages/torch/_dynamo/eval_frame.py", line 209, in _fn
    return fn(*args, **kwargs)
  File "repro.py", line 3, in fn
    def fn(x, y):
  File "python3.10/site-packages/torch/_dynamo/eval_frame.py", line 209, in _fn
    return fn(*args, **kwargs)
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 2812, in forward
    return compiled_fn(full_args)
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1222, in g
    return f(*args)
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1895, in runtime_wrapper
    all_outs = call_func_with_args(
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1247, in call_func_with_args
    out = normalize_as_list(f(args))
  File "/tmp/torchinductor/75/c75w2gc5qdvwzlouzm2j2qf2kow35v3nlkpbq2tiunw6kht3swv7.py", line 47, in call
    return (as_strided(buf0, (2, 2), (1, 2)), )
NameError: name 'buf0' is not defined
"""