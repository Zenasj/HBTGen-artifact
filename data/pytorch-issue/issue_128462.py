import torch.nn as nn

import torch
from torch.export import export
shape_or_input = [2,3]
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape_or_input[0] = x.shape[0]
        return torch.ops.aten.empty.memory_format(
            shape_or_input,
            dtype=torch.float32,
        )

inputs = [torch.randint(1, 3, shape_or_input, dtype=torch.int32)]
print(inputs)
empty_model = TestModule()

mod = export(empty_model, tuple(inputs))
out = mod.module()