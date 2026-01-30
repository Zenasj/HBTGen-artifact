import torch
from torch import _dynamo as dynamo

def f(a):
    return torch.softmax(a, dim=0)

def custom(gm, example_inputs):
    print("Recompiling")
    return gm.forward

def guard_fail(failure):
    print("Guard failure", failure)

f = dynamo.optimize(custom, dynamic=True, guard_fail_fn=guard_fail)(f)
f(torch.Tensor(2))
f(torch.Tensor(3))

valid_shape = a.ndim == 0 or py_all(a.shape[i] for i in dims)

valid_shape = a.ndim == 0 or py_all(a.shape[i] != 0 for i in dims)