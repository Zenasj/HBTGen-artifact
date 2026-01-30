import torch

m = MyScriptModule()
torch._C._jit_pass_constant_propagation(m.graph)