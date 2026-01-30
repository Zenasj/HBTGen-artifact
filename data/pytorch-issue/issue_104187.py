import torch
import torch.nn as nn

with torch.no_grad():
    for module in modules_to_materialize:
        # TODO: Emulate `module.to_empty(recurse=False)` until that is
        # explicitly supported
        named_params = [
            (param_name, param)
            for param_name, param in module.named_parameters(recurse=False)
        ]
        if len(named_params) == 0:
            continue
        for param_name, param in named_params:
            materialized_param = nn.Parameter(
                torch.empty_like(param, device=materialization_device)
            )
            delattr(module, param_name)
            setattr(module, param_name, materialized_param)
        module.reset_parameters()  # type: ignore[operator]