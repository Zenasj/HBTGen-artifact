import torch
import torch._functorch.config

torch._functorch.config.view_replay_for_aliased_outputs = True
torch._functorch.config.strict_autograd_cache = True

def f(a):
    tmp = a.detach()
    a.mul_(2)
    return a, tmp

with torch.autograd._force_original_view_tracking(True):
    fn = torch.compile(f)
    out = fn(torch.rand(2,3))

print(out)

import torch
import torch._functorch.config

torch._functorch.config.view_replay_for_aliased_outputs = True
torch._functorch.config.enable_autograd_cache = True
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.strict_autograd_cache = True

def f(a):
    tmp = a.detach()
    a.mul_(2)
    return a, tmp

with torch.autograd._force_original_view_tracking(True):
    fn = torch.compile(f)
    out = fn(torch.rand(2,3))

print(out)