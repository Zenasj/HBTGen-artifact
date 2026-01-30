import torch.nn as nn

import torch

x = torch.nn.Parameter(torch.empty(1, device="cuda"))
params_groups = [{
    "params": [x],
    "capturable": True
}]
optimizer = torch.optim.Adam(params_groups)
y = x ** 2
optimizer.zero_grad()
y.backward()
optimizer.step()

state['step'] = ( 
     torch.zeros((1,), dtype=torch.float, device=p.device) 
     if group['capturable'] or group['fused'] 
     else torch.tensor(0.) 
 )