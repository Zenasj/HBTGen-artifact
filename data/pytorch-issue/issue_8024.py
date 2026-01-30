import torch

traced_model = torch.jit.trace(input)(model)
fwd = traced_model._get_method('forward')
print(fwd.graph)