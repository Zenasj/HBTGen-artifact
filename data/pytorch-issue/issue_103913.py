import torch
from torch._dynamo.utils import counters
import json
from functools import wraps

counters.clear()

def foo(x, y):
    def add(x, y):
        return x + y

    @wraps(add)
    def wrapped_call(x, y):
        return add(x, y)

    return wrapped_call(x, y)

x = torch.ones(1,)
y = torch.ones(1,)

o = torch.compile(foo, fullgraph=False, backend='aot_eager')(x, y)

torch.testing.assert_close(o, 2 * x)
assert counters["graph_break"] == {
    "call_function wraps in skip_files /home/kshiteej/.conda/envs/pytorch-cuda-dev/lib/python3.9/functools.py": 1,
    "call_method UserDefinedObjectVariable(partial) __call__ [NestedUserFunctionVariable()] {}": 1
}
# print(json.dumps(counters, indent=2))

import torch
from torch._dynamo.utils import counters
import json
from functools import wraps

counters.clear()

def add(x, y):
        return x + y

def foo(x, y):
    @wraps(add)
    def wrapped_call(x, y):
        return add(x, y)

    return wrapped_call(x, y)

x = torch.ones(1,)
y = torch.ones(1,)

o = torch.compile(foo, fullgraph=False, backend='aot_eager')(x, y)

torch.testing.assert_close(o, 2 * x)
assert counters["graph_break"] == {}
# print(json.dumps(counters, indent=2))

import torch
from torch._dynamo.utils import counters
import json
from functools import wraps

counters.clear()

def foo(x, y):
    def add(x, y):
        return x + y

    @wraps(add)
    def wrapped_call(x, y):
        return add(x, y)

    return wrapped_call(x, y)

x = torch.ones(1,)
y = torch.ones(1,)

o = torch.compile(foo, fullgraph=True, backend='eager')(x, y)

torch.testing.assert_close(o, 2 * x)
print(counters["graph_break"])