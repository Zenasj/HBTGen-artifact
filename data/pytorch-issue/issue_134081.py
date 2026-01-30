import torch.nn as nn

import os
import torch

class UniqueTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        _x, _i = torch.unique(x, sorted=True, return_inverse=True)
        _x = _x.clone().detach()
        return self.relu(_x), _i


# test compile
with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UniqueTest().to(device=device)
    example_inputs=(torch.randn(8, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model.so")},
    )