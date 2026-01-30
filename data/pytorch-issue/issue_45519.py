import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Relu-Dropout JIT Test')
parser.add_argument('--bad-jit', action='store_true', help='Use new executor that cause Relu not to fuse with dropout.')
args = parser.parse_args()

assert hasattr(torch._C, '_jit_set_profiling_executor'), "Old JIT behavior doesn't exist!"
if not args.bad_jit :
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
else :
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)

@torch.jit.script
def jit_relu_dropout(x, prob) :
    # type: (Tensor, float) -> Tensor
    out = F.threshold(x, 0., 0.)
    out = F.dropout(out, p=prob, training=True)
    return out 
                                                                                    
inputs = torch.randn(5120, 1024, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

for x in range(0, 5) :
    outputs = jit_relu_dropout(inputs, 0.1)