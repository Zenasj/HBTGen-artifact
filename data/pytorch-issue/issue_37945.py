import torch.nn as nn

import torch
import torch.utils.mobile_optimizer
from torch.jit._recursive import wrap_cpp_module

class MyMod(torch.nn.Module):
    def forward(self, arg):
        return arg
    @torch.jit.export
    def version(self):
        return 1

sm = torch.jit.script(MyMod())
sm.eval()
print("Original")
sm._c.dump(True, True, False)
print("==========")
frozen = wrap_cpp_module(torch._C._freeze_module(sm_1._c))
print("POST FREEZING")
frozen._c.dump(True, True, False)
print("==========")