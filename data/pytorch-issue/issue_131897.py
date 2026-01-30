import torch.nn as nn

import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear")
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear")
        x = x + y
        return x


device = torch.device("cuda", 0)
model = MyModule().eval().to(device)

inputs = (torch.rand((1, 1, 32, 32), device=device), torch.rand((1, 1, 32, 32), device=device))

dim = torch.export.Dim("Dim", min=16, max=64)
dynamic_shapes = {"x": {2: dim, 3: dim}, "y": {2: dim, 3: dim}}

exported_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
print(exported_program)