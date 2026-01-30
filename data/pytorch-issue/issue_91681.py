import torchvision

import torch
from torchvision.models import resnet50

model = resnet50(weights=None)
compile_model = torch.compile(model)
print(type(compile_model))
example_forward_input = torch.rand(1, 3, 224, 224)
c_model_traced = torch.jit.trace(compile_model, example_forward_input) # or torch.jit.script
torch.jit.save(c_model_traced, "c_trace_model.pt")