import torch.nn as nn

import torch
 
class TestModule(torch.nn.Module):
    def forward(self, x):
        return x + torch.randn_like(x, device=x.device)
 
trace_input = torch.zeros([1], device='cuda', dtype=torch.float32)
module = torch.jit.trace_module(TestModule(), dict(forward=(trace_input,)), check_trace=False)
 
try:
    print('CUDA')
    module(trace_input)
    print('Correct\n')
except RuntimeError as e:
    print(e, '\n')
 
try:
    print('CPU')
    module(trace_input.cpu())
    print('Correct\n')
except RuntimeError as e:
    print(e, '\n')