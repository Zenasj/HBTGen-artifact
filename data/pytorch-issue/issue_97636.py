import torch.nn as nn

# ------------------------------------
# using torch.func (as of PyTorch 2.0)
# ------------------------------------
import torch
from torch.func import jacrev, functional_call
inputs = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

params = dict(model.named_parameters())
# jacrev computes jacobians of argnums=0 by default.
# We set it to 1 to compute jacobians of params
jacobians = jacrev(functional_call, argnums=1)(model, params, (inputs,))

jacrev(partial(functional_call, model), argnums=0)(params, input)

import torch
from functorch import jacrev

inputs = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

params = dict(model.named_parameters())

jacobians = jacrev(torch.nn.utils.stateless.functional_call, argnums=1)(model, params, (inputs,))
print(torch.__version__)  # 1.13.1+cu116