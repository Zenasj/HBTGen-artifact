python
import torch
import torch.nn as nn
import gc
from pytorch_memlab import MemReporter

model = nn.Sequential(nn.Linear(5, 4, bias=False), nn.Linear(4,5, bias=False)).cuda()
reporter = MemReporter(model)
model[0].register_full_backward_hook(lambda m, gin, gout: None)
x = torch.rand(7, 5).cuda()
y = model(x).sum()
y.backward()

reporter.report()
print([obj for obj in gc.get_objects() if isinstance(obj, torch.utils.hooks.BackwardHook)])