import torch
import torch.nn as nn

class MyParametrization(nn.Module):
    def forward(X):
        # This reparametrization just negates the original parameter value
        return -X

m = nn.Linear(...)
p = MyParametrization()
register_parametrization(m, "weight", p)

# Accessing the "weight" attribute will invoke p's forward() on m's original weight and return the output as the new weight.
# m.weight here is now an injected property that does the above instead of an actual Parameter.
# This property is defined in torch/nn/utils/parametrize.py.
m.weight

# NB: Parametrization changes the module type (e.g. torch.nn.utils.parametrize.ParametrizedLinear)
print(type(m))