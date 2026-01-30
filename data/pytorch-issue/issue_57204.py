import torch.nn as nn

import torch

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@torch.jit.script
def jit_relu_dropout(x, prob) :
    # type: (Tensor, float) -> Tensor
    x = torch.nn.functional.relu(x)
    x = torch.nn.functional.dropout(x, p=prob, training=True)
    return x

x = torch.randn((64, 40, 12, 1024), device="cuda:0", dtype=torch.float16, requires_grad=True)
y = jit_relu_dropout(x, 0.5)