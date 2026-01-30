import torch.nn as nn

opt.register_load_state_dict_pre_hook = disable(
                opt.register_load_state_dict_pre_hook
            )

import torch

@torch.compile(backend="eager")
def compiled_optim_init(params):
    optimizer = torch.optim.Adagrad(params)
    return optimizer

optimizer = compiled_optim_init([torch.nn.Parameter(torch.randn(2,3))])