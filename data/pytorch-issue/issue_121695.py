nn.Module.__call__

forward

forward_hook

import torch
import torch.nn as nn


class MyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buf0", torch.randn(10, 10))

    def forward(self, x):
        return x + self.buf0

def forward_hook(module: nn.Module, inputs: torch.Tensor, output: torch.Tensor):
    return output + 1

x = torch.randn(10, 10, device="cuda")
mod = MockModule().to("cuda")

mod.register_forward_hook(forward_hook)

opt_mod = torch.compile(backend="aot_eager", fullgraph=True)(mod)
y = opt_mod(x)
print(y)

TORCH_LOGS=+dynamo