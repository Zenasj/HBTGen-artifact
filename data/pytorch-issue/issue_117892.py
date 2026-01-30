import torch.nn as nn

import torch

class Bar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.ones(1))

    def forward(self, x):
        self.buf.add_(1)
        return x.sum() + self.buf.sum()

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.zeros(1))
        self.bar = Bar()

    def forward(self, x):
        self.buf.add_(1)
        bar = self.bar(x)
        self.bar.buf.add_(2)
        return bar.sum() + self.buf.sum()

tensor_input = torch.ones(5, 5)
exported_program = torch.export.export(Foo(), (tensor_input,))

dim0_x = torch.export.Dim("dim0_x")
# NOTE: If input is ExportedProgram, we need to specify dynamic_shapes
# as a tuple.
reexported_program = torch.export.export(
    exported_program, (tensor_input,), dynamic_shapes=({0: dim0_x},)
)
unflattened_module = torch.export.unflatten(reexported_program)
unflattened_module(tensor_input)