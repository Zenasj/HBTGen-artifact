import torch

def fn(x):
    # x: (2, 2, 3, 3, 2)
    v1_0 = torch.movedim(x, source=1, destination=2)
    v4_0 = x.add_(1)
    v0_0 = torch.cat([v4_0, v4_0], dim=2) # v0_0: (2, 2, 6, 3, 2)
    return [v1_0, v0_0]

x = torch.rand([2, 2, 3, 3, 2])

ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')

"""
==== Eager mode OK! ====
==== torchcomp compilation OK! ====
Traceback (most recent call last):
  File "repro.py", line 18, in <module>
    ret_compiled = compiled(x)
  File "python3.10/site-packages/torch/_dynamo/eval_frame.py", line 237, in _fn
    return fn(*args, **kwargs)
  File "/home/yuyao/bug_repro/repro.py", line 3, in fn
    def fn(v3_0):
  File "python3.10/site-packages/torch/_dynamo/eval_frame.py", line 237, in _fn
    return fn(*args, **kwargs)
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 3065, in forward
    return compiled_fn(full_args)
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1182, in g
    return f(*args)
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 2202, in runtime_wrapper
    regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
  File "python3.10/site-packages/torch/_functorch/aot_autograd.py", line 576, in gen_alias_from_base
    reshaped_base_tensor = aliased_base_tensor.as_strided(
RuntimeError: setStorage: sizes [2, 2, 6, 3, 2], strides [72, 36, 6, 2, 1], storage offset 0, and itemsize 4 requiring a storage size of 576 are out of bounds for storage of size 288
"""