import torch.nn as nn

import torch


def autocast_func_forward(orig_fwd):
    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    def new_fwd(*args, **kwargs):
        return orig_fwd(*args, **kwargs)

    return new_fwd

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear1.forward = autocast_func_forward(self.linear1.forward)

    def forward(self, x):
        return self.linear1(x)


x = torch.rand(10, 10)
m = BasicModule()
opt_m = torch.compile(backend="eager")(m)
res = opt_m(x)
print(res)

torch._dynamo.exc.Unsupported: 'inline in skipfiles: autocast_func_forward.<locals>.new_fwd | decorate_autocast /home/ybliang/local/pytorch/torch/amp/autocast_mode.py, skipped according skipfiles.SKIP_DIRS'

torch.autocast

torch.amp.autocast_mode.autocast

autocast_decorator