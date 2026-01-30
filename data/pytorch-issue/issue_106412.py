import torch.nn as nn

import torch
from torch._subclasses import fake_tensor
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._export import dynamic_dim

fake_mode = fake_tensor.FakeTensorMode(
    shape_env=ShapeEnv(  # all default values
        allow_scalar_outputs=True,
        allow_dynamic_output_shape_ops=True,
        assume_static_by_default=False,
    ),
)

class DynamicShapeSimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c) -> torch.Tensor:
        d = (torch.matmul(a, b) + c) / 2
        d_s0 = d.shape[0]
        d_s1 = d.shape[1]
        d_s3 = d_s0 * d_s1
        e = d.view(d_s3)
        return torch.cat([e, e])


with fake_mode:
    model = DynamicShapeSimpleModel()
    inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
    constraints = [
        dynamic_dim(inputs[0], 0),
        dynamic_dim(inputs[2], 0),
        dynamic_dim(inputs[2], 0) == dynamic_dim(inputs[0], 0),
    ]
    torch._dynamo.export(
        model,
        constraints=constraints,
        aten_graph=True,
    )(*inputs)