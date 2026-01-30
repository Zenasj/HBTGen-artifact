import gc
import torch
import torch.nn as nn
from torch.utils.module_tracker import ModuleTracker


class MyModel(nn.Module):
    def forward(self, x):
        return x * x

print(f"torch=={torch.__version__}")
m = MyModel()
m.cuda()
m.to(torch.bfloat16)
mt = ModuleTracker()
for i in range(1000):
    if i % 100 == 0:
        gc.collect()
        print("memory_allocated:", torch.cuda.memory_allocated())
    x = torch.randn([128, 256], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with mt:
        m(x)