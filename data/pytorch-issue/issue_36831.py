import torch.nn as nn

import torch

torch.manual_seed(0)
linear_parameters = list(torch.nn.Linear(1, 1).parameters())
print(linear_parameters)
optimizer = torch.optim.Adam(linear_parameters)
print(optimizer.state_dict())

# Optimizer.state_dict() relies on id(),
# so modification is required to remove the nondeterminism
param_mappings = {}
index = 0
for param_group in data['optimizer']['param_groups']:
    new_params = []
    for param in param_group['params']:
        param_mappings[param] = index
        new_params.append(index)
        index += 1
    param_group['params'] = new_params

new_state = {}
for k, v in data['optimizer']['state'].items():
    if k in param_mappings:
        new_k = param_mappings[k]
    else:
        new_k = k
    new_state[new_k] = v
data['optimizer']['state'] = new_state