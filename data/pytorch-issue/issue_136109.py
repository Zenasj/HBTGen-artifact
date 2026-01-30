import torch._custom_ops as custom_ops
import torch.utils.cpp_extension
from torch._custom_op.impl import CustomOp

test_ns = '_test_custom_op'
@custom_ops.custom_op(f'{test_ns}::foo')
def foo(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

custom_op = torch._custom_op.impl.get_op(f'{test_ns}::foo')
custom_ops._destroy(f'{test_ns}::foo')

x = torch.tensor([2])
custom_op(x)