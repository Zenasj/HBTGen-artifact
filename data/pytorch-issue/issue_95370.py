import torch

def fn(input):
    v = input.argmin(1) # v: (0, )
    return v.view([0, 3])

x = torch.rand([0, 3]) 
# if the shape is changed to [0, 1], torch.compile works fine.
# if we directly pass a tensor with shape [] to fn, torch.compile works fine.

ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')

"""
==== Eager mode OK! ====
==== torchcomp compilation OK! ====
[2023-02-23 03:22:03,608] torch._inductor.graph: [ERROR] Error from lowering
Traceback (most recent call last):
  File "python3.10/site-packages/torch/_inductor/ir.py", line 1389, in dynamic_reshape_indexer
    reindex = cls._dynamic_reshape_indexer(old_size, new_size)
  File "python3.10/site-packages/torch/_inductor/ir.py", line 1434, in _dynamic_reshape_indexer
    modulus = stack_old.pop()
IndexError: pop from empty list
"""