import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        out = torch.nn.functional.interpolate(x, size=(10, 20, 30), mode="trilinear", align_corners=True)
        return out + 1

# Build model, sample inputs, and validate model succeeds on sample inputs
model = MyModule().eval().cuda()
sample_input = torch.rand((1, 2, 3, 4, 5)).cuda()

model(sample_input)

# Try various export/tracing methods
try:
    torch._dynamo.export(model, sample_input, aten_graph=True, tracing_mode="symbolic")
except Exception as e:
    print("Dynamo export:\n", e)

try:
    torch._export.export(model, sample_input)
except Exception as e:
    print("Torch export:\n", e)

try:
    torch._functorch.aot_autograd.aot_export_module(model, sample_input, trace_joint=False)
except Exception as e:
    print("AOT export:\n", e)

try:
    torch.fx.experimental.proxy_tensor.make_fx(model, tracing_mode="symbolic", _allow_non_fake_inputs=True, pre_autograd=True)(sample_input)
except Exception as e:
    print("Make FX:\n", e)

print(torch._dynamo.explain(model, sample_input))