import torch.nn as nn

import torch

class DPER3ModuleInterface(torch.nn.Module):
    def __init__(self):
        super(DPER3ModuleInterface, self).__init__()

class DPER3ModuleList(DPER3ModuleInterface, torch.nn.ModuleList):
    def __init__(self, modules=None):
        DPER3ModuleInterface.__init__(self)
        torch.nn.ModuleList.__init__(self, modules)
        
class DPER3Sequential(DPER3ModuleInterface, torch.nn.Sequential):
    def __init__(self, modules=None):
        DPER3ModuleInterface.__init__(self)
        torch.nn.Sequential.__init__(self, modules)

class DPER3ModuleDict(DPER3ModuleInterface, torch.nn.ModuleDict):
    def __init__(self, modules=None):
        DPER3ModuleInterface.__init__(self)
        torch.nn.ModuleDict.__init__(self, modules)

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.submod = torch.nn.Linear(3, 4)
        self.modulelist = DPER3ModuleList([self.submod])
        self.sequential = DPER3Sequential(self.submod)
        self.moduledict = DPER3ModuleDict({"submod": self.submod})
    
    def forward(self, inputs):
        # ============== DPER3ModuleList ==============
        # Test `__getitem__()`
        assert self.modulelist[0] is self.submod
        
        # Test `__len__()`
        #
        # PROBLEM: this throws:
        #