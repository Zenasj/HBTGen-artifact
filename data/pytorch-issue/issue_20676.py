import torch.nn as nn

import torch
from torch import jit
from torch import nn
import numpy as np


class SubContainsVariable(jit.ScriptModule):
    def __init__(self):
        super(SubContainsVariable, self).__init__()
        self.v = nn.Parameter(torch.from_numpy(np.array(1.)))

    @jit.script_method
    def forward(self, x):
        return self.v * x


class ContainsVariable(jit.ScriptModule):
    def __init__(self):
        super(ContainsVariable, self).__init__()
        self.submodule = SubContainsVariable()
        self.submodule1 = jit.ScriptModule()
        self.submodule1.submodule2 = self.submodule
        self.v = nn.Parameter(torch.from_numpy(np.array(2.)))

    @jit.script_method
    def forward(self, x):
        return x + self.v + self.submodule(x) + self.submodule1.submodule2(x)


cv = ContainsVariable()
# both print True
print(cv.submodule is cv.submodule1.submodule2)
print(cv.submodule.v is cv.submodule1.submodule2.v)

jit.save(cv, "save/torchmodel.zip")
load = jit.load("save/torchmodel.zip")
# both print False
print(load.submodule is load.submodule1.submodule2)
print(load.submodule.v is load.submodule1.submodule2.v)