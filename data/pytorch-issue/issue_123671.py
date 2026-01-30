import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def forward(self, x, y):
        with torch.enable_grad():
            x = x + y
        return x

model = Model()
exported_program = torch.export._trace._export(
    model,
    (torch.tensor(2), torch.tensor(3)),
    dynamic_shapes=None,
    pre_dispatch=True,
    strict=False
)

def forward(self):
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False)