import torch.nn as nn

def my_load(self, state_dict, strict=True):
    self._orig_mod.load_state_dict(state_dict, strict)

model = torch.compile(model)
model.__class__.load_state_dict = my_load

import torch


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


module = M()
sd = module.state_dict()

compiled = torch.compile(module)
compiled.load_state_dict(sd)