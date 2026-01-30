import torch.nn as nn

import torch
from collections import OrderedDict
model_callbacks = torch.nn.ModuleDict(OrderedDict(
    lin_a=torch.nn.Linear(20, 5),
    lin_b=torch.nn.Linear(20, 10)
))
another_callbacks = torch.nn.ModuleDict()
another_callbacks.update(model_callbacks)

class ModuleDict(Module):
    def update(self, modules):
        # Some Code
        if isinstance(modules, container_abcs.Mapping): 
            # This condition will never be True if modules is nn.ModuleDict
            if isinstance(modules, (OrderedDict, ModuleDict)):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in sorted(modules.items()):
                    self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:  # Raise ValueError
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]

        # Other Code