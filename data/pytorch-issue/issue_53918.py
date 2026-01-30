import torch
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
def foo(x, y):
    return (x + y).relu()
m = torch.jit.script(foo)
x = torch.randn(65536).cuda().bfloat16()
y = torch.randn_like(x)
print(m.graph_for(x,y))