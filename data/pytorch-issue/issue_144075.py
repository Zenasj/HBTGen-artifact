import torch
import logging

torch._logging.set_logs(dynamo = logging.DEBUG)
torch._dynamo.reset()

device, backend = 'cpu', 'eager'

def fn(end):
    return torch.arange(start=0, step=2, end=end, device=device)

fn_cmp = torch.compile(fn, dynamic=None, fullgraph=True, backend=backend)

for end in [7,17,13]:
    res = fn_cmp(end)
    print(res)