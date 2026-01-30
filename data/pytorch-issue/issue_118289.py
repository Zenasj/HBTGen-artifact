import torch.nn as nn

import torch
from torch.export import export

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=32
            , kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (torch.randn(2, 3, 256, 256),)
const = torch.ones(2, 32, 256, 256)
example_kwargs = {"constant": const}
constraints = [dynamic_dim(example_args[0], 0), dynamic_dim(const, 0),
               dynamic_dim(const, 0) == dynamic_dim(example_args[0], 0),
               ]
exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs, constraints=constraints
  )
print(exported_program)