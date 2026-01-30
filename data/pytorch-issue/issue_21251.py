import torch.nn as nn
mod = nn.Module()
mod.add_module('a', nn.utils.spectral_norm(nn.Conv2d(2, 4, 3)))
state_dict = mod.state_dict()
mod.add_module('b', nn.utils.spectral_norm(nn.Conv2d(2, 4, 3)))
mod.load_state_dict(state_dict, strict=False)

IncompatibleKeys(missing_keys=['b.bias', 'b.weight_orig', 'b.weight_u', 'b.weight_v'], unexpected_keys=[])