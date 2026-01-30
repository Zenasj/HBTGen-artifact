import torch.nn as nn

import torch

def hook(self, *args, **kwargs):
    pass

module = torch.nn.Module()
module._register_load_state_dict_pre_hook(hook, with_module=True)
del module # there is still an owning reference in `module._load_state_dict_pre_hooks` hidden in closure made by `functools.partial`