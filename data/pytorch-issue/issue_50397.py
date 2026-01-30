import torch
import torch.fx

torch.fx.wrap('foo_wrapper')

def foo_wrapper(x):
    return torch.foo(x)


def to_trace(x):
    return foo_wrapper(x)

traced = torch.fx.symbolic_trace(to_trace)

scripted = torch.jit.script(traced)
"""
RuntimeError: 
object has no attribute foo:
  File "bad_compile_stack.py", line 7
def foo_wrapper(x):
    return torch.foo(x)
           ~~~~~~~~~ <--- HERE
'foo_wrapper' is being compiled since it was called from 'GraphModuleImpl.forward'
def forward(self, x):
    foo_wrapper = __main__.foo_wrapper(x);  x = None
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
    return foo_wrapper
"""