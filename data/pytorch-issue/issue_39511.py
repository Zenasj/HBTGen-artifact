import torch.nn as nn

import torch

flag = True
torch._C._jit_set_profiling_mode(flag)
torch._C._jit_set_profiling_executor(flag)
torch._C._jit_override_can_fuse_on_cpu(False)

def dropout_training(x):
    return torch.nn.functional.dropout(x)

x = torch.randn(100, 100, requires_grad=True, device="cuda")
scripted = torch.jit.script(dropout_training)
o = scripted(x)
o = scripted(x)
print("training graph: ", scripted.graph_for(x))