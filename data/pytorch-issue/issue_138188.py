import torch.nn as nn

import torch
model = torch.nn.Linear(1,2)
print(model.__class__.__name__)
compiled_model = torch.compile(model)
print(compiled_model.__class__.__name__)
print(compiled_model.training)
compiled_model.eval()
print(compiled_model.training)

Linear
OptimizedModule
True
True