import torch
import torch.nn as nn

from torch.testing._internal.common_utils import decorateIf
...

@decorateIf(unittest.skip, lambda params: params["x"] == 2)
@parametrize("x", range(5))
def test_foo(self, x):
    ...

@parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
@decorateIf(
    unittest.expectedFailure,
    lambda params: params["x"] == 3 and params["y"] == "baz"
)
def test_bar(self, x, y):
    ...

@decorateIf(
    unittest.expectedFailure,
    lambda params: params["op"].name == "add" and params["dtype"] == torch.float16
)
@ops(op_db)
def test_op_foo(self, device, dtype, op):
    ...

@decorateIf(
    unittest.skip,
    lambda params: params["module_info"].module_cls is torch.nn.Linear and \
        params["device"] == "cpu"
)
@modules(module_db)
def test_module_foo(self, device, dtype, module_info):
    ...