import torch.nn as nn

import torch
from torch.utils.cpp_extension import load_inline

# torch extensions cache should be cleared before the test
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Wrong env for the reproducer.")

class TestModel(torch.nn.Module):
    def forward(self, x):
        code = "int f() {return 2;}"
        module = load_inline(
                name='jit_extension',
                cpp_sources=code,
                functions='f',
                verbose=True)
        return x * module.f()


model = torch.nn.DataParallel(TestModel().cuda())
output = model(torch.ones([10, 1, 1, 1], device="cuda"))
assert torch.all(output == 2.)