import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = torch.nn.Conv2d(3, 3, 5, 2, 1)

    def forward(self, x):
        x = self.cv1(x)
        return x

x = torch.randn(10, 3, 20, 20) * 2
m = MyModule().eval()
x = x.cuda()
m = m.cuda()

with torch.no_grad():
    print("outside result: ", torch.jit.trace(m, x))
    with torch.cuda.amp.autocast(enabled = True, dtype=torch.float16):
        print("inside result: ", torch.jit.trace(m, x))

3
trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
    model, args, strict=False, _force_outplace=False, _return_inputs_states=True
)

3
prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
    # When weights are not reused, there is no perf impact
    # ONNX runtimes can also apply CSE optimization to compensate the lack of cache here
    torch.set_autocast_cache_enabled(False)
trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
    model, args, strict=False, _force_outplace=False, _return_inputs_states=True
)
torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)

with torch.no_grad():
    print("outside result: ", torch.jit.script(m, x))
    with torch.cuda.amp.autocast(enabled = True, dtype=torch.float16):
        print("inside result: ", torch.jit.script(m, x))