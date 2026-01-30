import torch.nn as nn

import io
import torch
import torch.utils.collect_env

torch.utils.collect_env.main()

class MyMod(torch.nn.Module):
    def forward(self, arg):
        return torch.nn.functional.interpolate(
            arg,
            scale_factor=2.0,
            recompute_scale_factor=False)

scripted = torch.jit.script(MyMod())
print()
print(scripted.code)
print(scripted.graph)
print()
print(scripted(torch.zeros(1,1,1,1)))