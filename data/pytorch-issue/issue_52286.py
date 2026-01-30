import torch

torch._C._jit_set_profiling_executor(False)
model = torch.jit.script(model)
model = torch.jit.freeze(model)
model = torch.jit.optimize_for_inference(model)