import torch.nn as nn

import torch

from collections import OrderedDict


DEVICE = "cpu"

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ZeROOrderedDict(OrderedDict):
    def __init__(self, parent_module=None, *args, **kwargs):
        """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

        Args:
            parent_module (``collections.OrderedDict``): the collection to replace
        """

        super().__init__(*args, **kwargs)
        self._parent_module = parent_module

    def __getitem__(self, key):
        param = super().__getitem__(key)

        # Params can be registered as None (e.g., bias)
        if param is None:
            return param
        
        # do something here
        return param


def inject_parameters(module, cls):
    for module in module.modules():
        if cls == ZeROOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()

        for key, param in module._parameters.items():
            new_param[key] = param
        module._parameters = new_param


model = M()


inject_parameters(model, ZeROOrderedDict)

model= model.to(DEVICE)

model = torch.compile(model, backend="inductor", fullgraph=False)

x = torch.ones(2).to(DEVICE)
with torch.no_grad():
    y = model(x)
print(y)