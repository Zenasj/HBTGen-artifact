import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, x, y, z):
        a = y.shape[0]
        b = z.shape[0]

        def true_fn(x):
            return x + a

        def false_fn(x):
            return x + b * z

        return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

input1 = (torch.ones(3, 3), torch.ones(5), torch.ones(3, 3))
model = M().cuda()
dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}

ep = torch.export.export(model, input1, dynamic_shapes=dynamic_shapes, strict=False)