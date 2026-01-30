import torch.nn as nn

python

import os
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        return x
 
with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs=(torch.randn(2, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)

    ######################################################
    exmp_inputs=torch.randn(2, 10)
    ep = torch.export.export(model.eval(), example_inputs)

    from torch._decomp import get_decompositions
    decomp_tbl = get_decompositions([torch.ops.aten.linear.default])
    exp_mod = ep.run_decompositions(decomp_tbl)

    model_out = op_replacement(exp_mod.module())

    so_path = torch._export.aot_compile(
        model_out,
        example_inputs,
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model_addmm.so")},
    )