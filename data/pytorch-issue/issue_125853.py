import torch

from torch._subclasses.fake_tensor import FakeTensorMode 
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv, DimDynamic, StatelessSymbolicContext
)

shape_env = ShapeEnv()
t1 = torch.ones(2, 2, 768)
with FakeTensorMode(shape_env=shape_env) as fake_mode:
    t = fake_mode.from_tensor(
        t1,
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC, DimDynamic.STATIC],
        )
    )
print(t)  # FakeTensor(..., size=(s0, 2, 768))
print(torch.ops.aten.unsqueeze(t, 1))  # FakeTensor(..., size=(s0, 1, 2, 768))
print(torch.ops.aten.unsqueeze_copy(t, 1))  # FakeTensor(..., size=(2, 1, 2, 768))