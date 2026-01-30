import torch

def fn(input):
    return input.sub(1, alpha=2) # ❌
    # input.sub(1) ✅
    # input.sub(torch.tensor([1]), alpha=2) ✅

x = torch.rand([1], dtype=torch.float64) # float16, float32 ✅

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
  File "python3.10/site-packages/torch/_dynamo/utils.py", line 1196, in run_node
    return getattr(args[0], node.target)(*args[1:], **kwargs)
  File "python3.10/site-packages/torch/utils/_stats.py", line 20, in wrapper
    return fn(*args, **kwargs)
  File "python3.10/site-packages/torch/_subclasses/fake_tensor.py", line 989, in __torch_dispatch__
    return self.dispatch(func, types, args, kwargs)
  File "python3.10/site-packages/torch/_subclasses/fake_tensor.py", line 1172, in dispatch
    r = func(*args, **kwargs)
  File "python3.10/site-packages/torch/_ops.py", line 284, in __call__
    return self._op(*args, **kwargs or {})
  File "python3.10/site-packages/torch/_prims_common/wrappers.py", line 220, in _fn
    result = fn(*args, **kwargs)
  File "python3.10/site-packages/torch/_prims_common/wrappers.py", line 130, in _fn
    result = fn(**bound.arguments)
  File "python3.10/site-packages/torch/_refs/__init__.py", line 1608, in sub
    return prims.sub(a, b)
  File "python3.10/site-packages/torch/_ops.py", line 284, in __call__
    return self._op(*args, **kwargs or {})
  File "python3.10/site-packages/torch/_prims/__init__.py", line 341, in _elementwise_meta
    utils.check_same_dtype(*args)
  File "python3.10/site-packages/torch/_prims_common/__init__.py", line 1079, in check_same_dtype
    raise RuntimeError(msg)
RuntimeError: Tensor with dtype torch.float32 is not the expected dtype of torch.float64!
"""