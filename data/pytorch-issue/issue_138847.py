import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, x):
        if x.size(0) % 2 == 0:
            return x.clone() * 2
        else:
            return x.clone() * 0

input1 = (torch.rand(size=(4,), device="cuda"),)
model = M().cuda()

_ = model(*input1)

dynamic_shapes = {
    "x": {0: torch.export.Dim.DYNAMIC},
}
ep = torch.export.export(model, input1, dynamic_shapes=dynamic_shapes, strict=False)