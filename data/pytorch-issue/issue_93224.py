import torch

def fn(x):
    return torch.bitwise_xor(x, 18)
    # return torch.bitwise_xor(x, torch.tensor([18])) # works fine

x = torch.rand(2).int()
o = fn(x)
print('==== Eager mode OK! ====')
compiled = torch.compile(fn, fullgraph=True)
print(compiled(x))
print('==== compile mode OK! ====')

import torch

def fn(x):
    # Dispatcher:
    # [call] op=[aten::add.Tensor], key=[PythonTLSSnapshot]
    #   [redispatchBoxed] op=[aten::add.Tensor], key=[AutogradCPU]
    #    [redispatch] op=[aten::add.Tensor], key=[Python]
    #     [callBoxed] op=[aten::add.Tensor], key=[Meta]
    #      [callBoxed] op=[prims::add], key=[Meta]
    #       [call] op=[aten::empty_strided], key=[BackendSelect]
    #        [redispatch] op=[aten::empty_strided], key=[Meta]
    #     [call] op=[aten::detach], key=[Meta]
    torch.add(x, 18)  # works

    # torch.lt(x, 18)  # works

    # Dispatcher:
    # [call] op=[aten::bitwise_xor.Scalar], key=[PythonTLSSnapshot]
    #   [redispatchBoxed] op=[aten::bitwise_xor.Scalar], key=[AutogradCPU]
    #    [call] op=[aten::bitwise_xor.Tensor], key=[PythonTLSSnapshot]
    #     [redispatchBoxed] op=[aten::bitwise_xor.Tensor], key=[AutogradCPU]
    #      [redispatchBoxed] op=[aten::bitwise_xor.Tensor], key=[Python]  # Errors!
    torch.bitwise_xor(x, 18)

from torch._subclasses.fake_tensor import (
    FakeTensorMode,
)

with FakeTensorMode():
    x = torch.ones(2, device=torch.device('cpu'), dtype=torch.int)
    fn(x)

# torch.ops.add.Tensor -> refs.add -> prims.add

# torch.ops.bitwise_xor.Tensor - Crash

from torch._subclasses.fake_tensor import (
    FakeTensorMode,
)
from torch.testing._internal.common_methods_invocations import op_db, BinaryUfuncInfo

binary_ops = list(filter(lambda op: isinstance(op, BinaryUfuncInfo), op_db))

for op in binary_ops:
    if op.supports_rhs_python_scalar:
        try:
            with FakeTensorMode():
                x = torch.ones(2, device=torch.device('cpu'), dtype=torch.int)
                op(x, 18)
        except Exception as e:
            if "Expected a value of type 'Tensor' for argument" in str(e):
                print(op.name)
            else:  # If different error from expected
                print(op.name)
                print(e)
                print()
            continue

    if op.supports_one_python_scalar:
        try:
            with FakeTensorMode():
                x = torch.ones(2, device=torch.device('cpu'), dtype=torch.int)
                op(18, x)
        except Exception as e:
            if "Expected a value of type 'Tensor' for argument" in str(e):
                print(op.name)
            else:  # If different error from expected
                print(op.name)
                print(e)
                print()
            continue
    
    if op.supports_two_python_scalars:
        try:
            with FakeTensorMode():
                op(18, 18)
        except Exception as e:
            if "Expected a value of type 'Tensor' for argument" in str(e):
                print(op.name)
            else:  # If different error from expected
                print(op.name)
                print(e)
                print()
            continue