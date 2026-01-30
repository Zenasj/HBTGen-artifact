import torch
import torch.nn as nn

from torch import nn
from torch.nn.modules.container import ModuleDict
from torch.distributed._sharded_tensor import state_dict_hook


class LitAutoEncoder(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.num_dict = depth

        def recursive_build_encoder(stop_num, current_num):
            if stop_num == current_num:
                return nn.Sequential(nn.Linear(current_num + 1, current_num + 1))
            else:
                module_dict = ModuleDict()
                module_dict[str(current_num)] = recursive_build_encoder(stop_num, current_num + 1)
                return module_dict

        self.encoder = recursive_build_encoder(depth, 0)

        # register state dict hook
        self._register_state_dict_hook(state_dict_hook)


# this does not terminate in a reasonable time!
enc = LitAutoEncoder(depth=100)
enc.state_dict()

def _recurse_update_dict(module, destination, prefix):
    for submodule_name, submodule in module.named_modules():
        for attr_name, attr in module.__dict__.items():
            if isinstance(attr, ShardedTensor):
                destination[prefix + submodule_name + "." + attr_name] = attr