import torch

compiled_1 = torch.compile("max-autotune", fullgraph=True)(fn)
# run compiled_1
out_compiled = compiled_1
compiled_2 = torch.compile("max-autotune", fullgraph=True)(fn2)