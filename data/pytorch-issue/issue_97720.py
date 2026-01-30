import torch.nn as nn

import torch
class FeatureDict(torch.nn.Module):
    #Torch ModuleDict wrapper that permits keys with any name.
    def __init__(self):
        super().__init__()
        self.module_dict = torch.nn.ModuleDict()
    def __getitem__(self, key) -> torch.nn.Module:
        return self.module_dict[key]
    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        self.module_dict[key] = module

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ld = FeatureDict()
        
        self.ld["key1"] = torch.nn.Linear(1,1)
        
    def forward(self, x):
        return self.ld["key1"](x)

foo = Foo()
cfoo = torch.compile(foo)
cfoo(1)