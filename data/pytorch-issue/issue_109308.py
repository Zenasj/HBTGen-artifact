import torch
from functorch.experimental import control_flow
def exportdb_example2(x):
    def true_fn():
        return torch.sin(x)

    def false_fn():
        return torch.cos(x)

    return control_flow.cond(x.sum() > 0, true_fn, false_fn, [])
ep = torch._export.export(exportdb_example2, (torch.randn(4, 5),))