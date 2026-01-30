import torch.nn as nn

import torch  # torch version is 1.11.0

def bp_hook(module, grad_input, grad_output):
    param_grad = list(module.parameters())[0].grad
    print(f'gradient of the module inside bp_hook: {param_grad}')

net = torch.nn.Linear(4, 1, bias=False)
net.register_full_backward_hook(bp_hook)
data = torch.ones(1, 4)
output = net(data)
output.backward()
print(f'gradient of the module outside bp_hook: {list(net.parameters())[0].grad}')