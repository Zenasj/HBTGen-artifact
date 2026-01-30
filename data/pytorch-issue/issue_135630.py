import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def forward(self, x):
        return bool(x.eq(0.1).any())

model = Foo()
inputs = (torch.randn(64),)
with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
    ep = export(model, inputs, strict=False)