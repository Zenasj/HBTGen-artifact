import torch.nn as nn

import torch
from torch import _dynamo as dynamo

class DynamicSliceExportMod(torch.nn.Module):
    def forward(self, x):
        results = []
        for i in range(4):
            results.append(x[: x.size(0) - i, i : x.size(2), i:3])
        return tuple(results)

x = torch.rand(5, 5, 5)

gm_symbolic, _ = dynamo.export(DynamicSliceExportMod(), x, aten_graph=True, tracing_mode="symbolic")
gm_symbolic.print_readable()

gm_fake, _ = dynamo.export(DynamicSliceExportMod(), x, aten_graph=True, tracing_mode="fake")
gm_fake.print_readable()

gm_real, _ = dynamo.export(DynamicSliceExportMod(), x, aten_graph=True, tracing_mode="real")
gm_real.print_readable()

gm_whatever, _ = dynamo.export(DynamicSliceExportMod(), x, aten_graph=True, tracing_mode="whatever")
gm_whatever.print_readable()

import torch
from torch import _dynamo as dynamo
from torch.fx.experimental import proxy_tensor

class DynamicSliceExportMod(torch.nn.Module):
    def forward(self, x):
        results = []
        for i in range(4):
            results.append(x[: x.size(0) - i, i : x.size(2), i:3])
        return tuple(results)

x = torch.rand(5, 5, 5)

gm_symbolic, _ = dynamo.export(DynamicSliceExportMod(), x, aten_graph=True, tracing_mode="symbolic")
gm_symbolic.print_readable()
dynamo.reset()

opt_gm = None
def func(gm, _):
    global opt_gm
    opt_gm = gm
    return gm

dynamo.optimize(func, dynamic=True)(DynamicSliceExportMod())(x)
opt_gm = proxy_tensor.make_fx(opt_gm, tracing_mode="symbolic")(x)
opt_gm.print_readable()