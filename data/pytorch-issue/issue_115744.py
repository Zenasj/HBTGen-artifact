import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
torch.manual_seed(0)

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

x = torch.randn(3)

def f(x):
    layer = nn.Linear(3, 3, bias=False)
    parametrize.register_parametrization(layer, "weight", Symmetric())
    y = layer(x)
    return y, layer.weight

f_compiled = torch.compile(f, backend="aot_eager")

out_compiled, weight = f_compiled(x)
out = torch.nn.functional.linear(x, weight)
assert torch.allclose(out_compiled, out)