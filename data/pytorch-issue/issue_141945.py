import torch.nn as nn

# import os
# os.environ["TORCH_LOGS"] = "+dynamo"

import torch
# torch._dynamo.config.inline_inbuilt_nn_modules = False
import enum
from collections import OrderedDict

DEVICE = "cpu"

class ParametersModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.scale = torch.nn.Parameter(torch.randn(2, 2))
        self.scale_dup = self.scale

    def forward(self, x):
        counter = 0
        for param in self.parameters():
            counter += 1
        return x * self.scale * counter


class ZeroParamStatus(enum.Enum):
    # parameters are fully present and ready for use on all processes
    AVAILABLE = 1

    # parameters are either partitioned or remote in some or all process
    NOT_AVAILABLE = 2

    # parameters are being gathered.
    INFLIGHT = 3


class ZeROOrderedDict(OrderedDict):
    def __init__(self, parent_module=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent_module = parent_module

    def __getitem__(self, key):
        param = super().__getitem__(key)

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            pass

        return param


def inject_parameters(module, cls):
    for module in module.modules():
        if cls == ZeROOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()

        for key, param in module._parameters.items():
            # just a hack to set the status
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            new_param[key] = param
        module._parameters = new_param


model = ParametersModule()
inject_parameters(model, ZeROOrderedDict)

model= model.to(DEVICE)
model = torch.compile(model, backend="inductor", fullgraph=False)
x = torch.ones(2).to(DEVICE)
y = model(x)
print(y)