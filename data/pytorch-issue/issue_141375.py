import torch.nn as nn

from tensordict import from_module, TensorDictParams, TensorDict
import torch.nn

module = torch.nn.Module()
module.params = torch.nn.Parameter(torch.ones(3))
params2 = TensorDict({"params": torch.ones(3) + 1})

@torch.compile(fullgraph=True)
def func(z, params2):
    with params2.to_module(module):
        out = z + module.params
    return out 

print(func(torch.zeros(()), params2))