import torch

from castle.algorithms import *
model = DAG_GNN(device_type='gpu', batch_size=3500, device_ids=0)
model.learn(x)

state['step'] = (
    torch.zeros((), dtype=torch.float, device=p.device)
    if group['capturable'] or group['fused']
    else torch.tensor(0., device=p.device)
)
print(state['step'].dtype)
exit(0)

torch.float64